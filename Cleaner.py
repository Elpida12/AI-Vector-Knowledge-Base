#!/usr/bin/env python3
"""
Purpose: Quality first, Incremental, and avoids duplications on reruns
- Keeps only high-quality chunks (syntax, optional docstrings, optional Ruff clean)
- Persists a hash index in SQLite, so other runs never create duplicates again
- Skips unchanged source files using a file-state cache (mtime/size or sha256)
- Can "seed" known chunk hashes from previous manifests or by scanning the output directory
- Includes an optional audit summary 
- Black/Ruff optional
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import tokenize
from multiprocessing import Pool, cpu_count
from functools import partial    
import black

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------------------------
# Utilities
# ---------------------------

def detab_leading(text: str, tab_width: int = 4) -> str:
    if "\t" not in text:
        return text
    out_lines: list[str] = []
    tw = max(1, int(tab_width))
    for line in text.splitlines(True):
        m = re.match(r"^(\s*)", line)
        indent = m.group(1) if m else ""
        indent_detabbed = indent.replace("\t", " " * tw)
        out_lines.append(indent_detabbed + line[len(indent):])
    return "".join(out_lines)


def naive_strip_line_comments(src: str) -> str:
    out: list[str] = []
    for line in src.splitlines(True):
        i = 0
        in_sq = False
        in_dq = False
        esc = False
        while i < len(line):
            ch = line[i]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "#" and not in_sq and not in_dq:
                line = line[:i] + ("\n" if line.endswith("\n") else "")
                break
            i += 1
        out.append(line)
    return "".join(out)


# ---------------------------
# Comment / Docstring stripping
# ---------------------------

def strip_comments_only(src: str, tab_width: int, on_error: str) -> str:
    pre = detab_leading(src, tab_width=tab_width)
    buff = io.StringIO(pre)
    out_tokens = []
    try:
        for tok in tokenize.generate_tokens(buff.readline):
            tok_type, tok_str, start, end, line = tok
            if tok_type == tokenize.COMMENT:
                continue
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except (tokenize.TokenError, IndentationError, TabError):
        if on_error == "fallback":
            return naive_strip_line_comments(src)
        elif on_error == "keep":
            return src
        else:
            raise


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, body: List[ast.stmt]) -> List[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:  
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST: 
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node


def remove_docstrings(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree) 
    except Exception:
        return src


# ---------------------------
# Complexity & trivial checks
# ---------------------------

def cyclomatic_complexity(node: ast.AST) -> int:
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp, ast.Match)):
            c += 1
        elif isinstance(n, (ast.comprehension, ast.ExceptHandler)):
            c += 1
    return c


def is_trivial_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = fn.body or []
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        rest = body[1:]
    else:
        rest = body
    if not rest:
        return True
    if len(rest) == 1 and isinstance(rest[0], (ast.Pass,)):
        return True
    if len(rest) == 1 and isinstance(rest[0], ast.Return):
        val = rest[0].value
        if val is None:
            return True
        if isinstance(val, ast.Constant) and val.value in (None, Ellipsis):
            return True
    if len(rest) == 1 and isinstance(rest[0], ast.Expr) and isinstance(getattr(rest[0], "value", None), ast.Constant) and rest[0].value.value is Ellipsis:
        return True
    return False


# ---------------------------
# Formatting and hashing
# ---------------------------

def format_with_black(code: str) -> str:
    if not code.strip():
        return code
    if black is None:
        return code
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return code


def stable_hash(code: str) -> str:
    norm = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()


# ---------------------------
# Ruff integration (optional)
# ---------------------------

def ruff_available() -> bool:
    from shutil import which
    return which("ruff") is not None


def ruff_check(code: str, select: str = "E,F") -> tuple[bool, str]:
    if not ruff_available():
        return True, ""
    try:
        proc = subprocess.run(
            ["ruff", "check", "-", "--select", select, "--quiet"],
            input=code.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=500,  #  prevent hangs
        )
        out = proc.stdout.decode("utf-8", "ignore") + proc.stderr.decode("utf-8", "ignore")
        ok = proc.returncode == 0
        return ok, out.strip()
    except subprocess.TimeoutExpired:
        return False, "ruff timeout (>45s)"
    except Exception as e:
        return True, f"ruff error: {e}"

# ---------------------------
# SQLite state (persistent)
# ---------------------------

class StateDB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), timeout=30)   
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")       #  wait up to 5s on lock
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
              path TEXT PRIMARY KEY,
              mtime REAL,
              size INTEGER,
              sha256 TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              hash TEXT PRIMARY KEY,
              out_path TEXT,
              added_at REAL
            )
            """
        )
        self.conn.commit()

    # ---- files ----
    def get_file(self, path: Path) -> Optional[tuple[float,int,str]]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT mtime,size,sha256 FROM files WHERE path=?", (str(path),)).fetchone()
        if row:
            return float(row[0]), int(row[1]), (row[2] or "")
        return None

    def upsert_file(self, path: Path, mtime: float, size: int, sha256: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO files(path,mtime,size,sha256) VALUES(?,?,?,?)\n             ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime,size=excluded.size,sha256=excluded.sha256",
            (str(path), mtime, size, sha256),
        )
        self.conn.commit()

    # ---- chunks ----
    def has_chunk(self, h: str) -> bool:
        cur = self.conn.cursor()
        row = cur.execute("SELECT 1 FROM chunks WHERE hash=?", (h,)).fetchone()
        return row is not None

    def add_chunk(self, h: str, out_path: Path) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO chunks(hash,out_path,added_at) VALUES(?,?,?)",
            (h, str(out_path), time.time()),
        )
        self.conn.commit()

    def preload_hashes_from_manifest(self, manifest_paths: list[Path]) -> int:
        imported = 0
        for mp in manifest_paths:
            try:
                with mp.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        h = (row.get("sha256") or row.get("hash") or "").strip()
                        outp = (row.get("chunk_file") or row.get("out_path") or "").strip()
                        if len(h) == 64:
                            self.add_chunk(h, Path(outp or f"manifest:{mp.name}"))
                            imported += 1
            except Exception:
                pass
        return imported

    def preload_hashes_by_scanning_output(self, output_dir: Path) -> int:
        imported = 0
        for p in output_dir.rglob("*.py"):
            try:
                code = p.read_text(encoding="utf-8")
            except Exception:
                continue
            h = stable_hash(code)
            if not self.has_chunk(h):
                self.add_chunk(h, p)
                imported += 1
        return imported

    def close(self):
        self.conn.close()


# ---------------------------
# Core pipeline
# ---------------------------

def collect_def_nodes(module: ast.Module, include_nested: bool) -> list[tuple[str, str, int, int]]:
    nodes: list[tuple[str, str, int, int]] = []

    def add(n: ast.AST, kind: str, name: str):
        lineno = getattr(n, "lineno", None)
        end_lineno = getattr(n, "end_lineno", None)
        if isinstance(lineno, int) and isinstance(end_lineno, int):
            nodes.append((kind, name, lineno, end_lineno))

    it: Iterable[ast.AST] = ast.walk(module) if include_nested else module.body

    for n in it:
        if isinstance(n, ast.FunctionDef):
            add(n, "function", n.name)
        elif isinstance(n, ast.AsyncFunctionDef):
            add(n, "async_function", n.name)
        elif isinstance(n, ast.ClassDef):
            add(n, "class", n.name)

    nodes.sort(key=lambda t: (t[2], t[0], t[1]))
    return nodes


def gate_and_emit_chunk(
    code: str,
    node_kind: str,
    node_name: str,
    start: int,
    end: int,
    output_prefix: Path,
    min_lines: int,
    max_lines: int,
    require_docstrings: bool,
    use_ruff: bool,
    ruff_enforce: bool,
    state: StateDB,
    counters: Counter,
) -> Optional[Path]:
    counters["chunks.total_candidates"] += 1

    code = format_with_black(code)

    try:
        nmod = ast.parse(code)
    except SyntaxError:
        counters["chunks.parse_fail"] += 1
        return None

    has_doc = False
    cc = 1
    is_trivial = False
    target_node = None
    for n in nmod.body:
        if node_kind in ("function", "async_function") and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            target_node = n
            break
        if node_kind == "class" and isinstance(n, ast.ClassDef):
            target_node = n
            break
    if target_node is not None:
        has_doc = ast.get_docstring(target_node) is not None
        cc = cyclomatic_complexity(target_node)
        if isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_trivial = is_trivial_function(target_node)

    line_count = code.count("\n") + (1 if code and not code.endswith("\n") else 0)
    if line_count < min_lines:
        counters["chunks.line_too_small"] += 1
        return None
    if max_lines > 0 and line_count > max_lines:
        counters["chunks.line_too_large"] += 1
        return None

    if require_docstrings and not has_doc and node_kind in ("function", "async_function", "class"):
        counters["chunks.no_docstring"] += 1
        return None

    if is_trivial:
        counters["chunks.trivial"] += 1
        return None

    ruff_ok = True
    ruff_notes = ""
    if use_ruff:
        ruff_ok, ruff_notes = ruff_check(code)
        if ruff_enforce and not ruff_ok:
            counters["chunks.ruff_fail"] += 1
            return None

    h = stable_hash(code)
    if state.has_chunk(h):
        counters["chunks.duplicate_global"] += 1
        return None

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(f"{output_prefix}_{node_kind}_{node_name}_{h[:8]}.py")
    out_path.write_text(code, encoding="utf-8")

    state.add_chunk(h, out_path)

    counters["chunks.emitted"] += 1

    return out_path


# -------- I/O helpers --------

def read_text_any(path: Path, errors: str = "strict") -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors=errors)
    except Exception:
        if errors == "strict":
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                return None
        return None


def sanitized_rel_component(path_under_input: Path) -> str:
    rel_str = str(path_under_input).strip().strip("/\\")
    if not rel_str:
        return "root"
    return re.sub(r"[\\/]+", "_", rel_str)


# --------------
# Walk & process
# --------------

def process_file(
    file_path: Path,
    input_dir: Path,
    output_dir: Path,
    keep_docstrings: bool,
    top_level_only: bool,
    max_comment_ratio: float,
    errors_policy: str,
    tab_width: int,
    on_tok_err: str,
    min_lines: int,
    max_lines: int,
    require_docstrings: bool,
    use_ruff: bool,
    ruff_enforce: bool,
    state: StateDB,
    counters: Counter,
    min_chars_legacy: int,
) -> None:
    counters["files.seen"] += 1

    # Fast skip unchanged files using mtime/size 
    st = file_path.stat()
    rec = state.get_file(file_path)
    if rec and rec[0] == st.st_mtime and rec[1] == st.st_size:
        counters["files.unchanged_skip"] += 1
        return

    raw = read_text_any(file_path, errors=errors_policy)
    if raw is None:
        counters["files.read_fail"] += 1
        print(f"[skip] could not read: {file_path}")
        return

    try:
        no_comments = strip_comments_only(raw, tab_width=tab_width, on_error=on_tok_err)
    except Exception:
        counters["files.tokenize_fail"] += 1
        print(f"[skip] tokenize failed: {file_path}")
        # update file state anyway to avoid re-reading repeatedly
        state.upsert_file(file_path, st.st_mtime, st.st_size, hashlib.sha256(raw.encode('utf-8','ignore')).hexdigest())
        return

    cleaned = no_comments if keep_docstrings else remove_docstrings(no_comments)

    if max_comment_ratio >= 0:
        ratio = max(0, len(raw) - len(cleaned)) / max(1, len(raw))
        if ratio > max_comment_ratio:
            counters["files.high_comment_ratio"] += 1
            state.upsert_file(file_path, st.st_mtime, st.st_size, hashlib.sha256(raw.encode('utf-8','ignore')).hexdigest())
            return

    try:
        mod = ast.parse(cleaned)
    except SyntaxError:
        counters["files.syntax_error_after_clean"] += 1
        state.upsert_file(file_path, st.st_mtime, st.st_size, hashlib.sha256(raw.encode('utf-8','ignore')).hexdigest())
        return

    defs = collect_def_nodes(mod, include_nested=not top_level_only)
    if not defs:
        counters["files.no_defs"] += 1
        state.upsert_file(file_path, st.st_mtime, st.st_size, hashlib.sha256(raw.encode('utf-8','ignore')).hexdigest())
        return

    rel = file_path.parent.relative_to(input_dir)
    repo_part = sanitized_rel_component(rel)
    save_prefix = output_dir / f"{repo_part}_{file_path.stem}"

    lines = cleaned.splitlines()

    emitted_any = False
    for (kind, name, start, end) in defs:
        start_i = max(0, start - 1)
        end_i = max(start_i, end)
        chunk = "\n".join(lines[start_i:end_i])
        if len(chunk.strip()) < max(0, min_chars_legacy):
            continue
        outp = gate_and_emit_chunk(
            code=chunk,
            node_kind=kind,
            node_name=name,
            start=start,
            end=end,
            output_prefix=save_prefix,
            min_lines=min_lines,
            max_lines=max_lines,
            require_docstrings=require_docstrings,
            use_ruff=use_ruff,
            ruff_enforce=ruff_enforce,
            state=state,
            counters=counters,
        )
        if outp is not None:
            emitted_any = True

    # Update file state once processed
    raw_sha = hashlib.sha256(raw.encode('utf-8','ignore')).hexdigest()
    state.upsert_file(file_path, st.st_mtime, st.st_size, raw_sha)

    if emitted_any:
        counters["files.processed_emitted"] += 1
    else:
        counters["files.processed_no_emit"] += 1


_STATE = None  # process-local StateDB
def _worker_init(state_db_path: str) -> None:  
    # Create a process-local SQLite connection 
    global _STATE
    _STATE = StateDB(Path(state_db_path).resolve())

def _worker_process_one(
    file_path_str: str,
    input_dir_str: str,
    output_dir_str: str,
    keep_docstrings: bool,
    top_level_only: bool,
    max_comment_ratio: float,
    errors_policy: str,
    tab_width: int,
    on_tok_err: str,
    min_lines: int,
    max_lines: int,
    require_docstrings: bool,
    use_ruff: bool,
    ruff_enforce: bool,
    min_chars_legacy: int,
) -> dict:
    # Run the existing process_file on a single path using the process-local StateDB. Return the counters so the parent can aggregate.
    assert _STATE is not None, "Worker StateDB not initialized"
    counters = Counter()
    try:
        process_file(
            file_path=Path(file_path_str),
            input_dir=Path(input_dir_str),
            output_dir=Path(output_dir_str),
            keep_docstrings=keep_docstrings,
            top_level_only=top_level_only,
            max_comment_ratio=max_comment_ratio,
            errors_policy=errors_policy,
            tab_width=tab_width,
            on_tok_err=on_tok_err,
            min_lines=min_lines,
            max_lines=max_lines,
            require_docstrings=require_docstrings,
            use_ruff=use_ruff,
            ruff_enforce=ruff_enforce,
            state=_STATE,
            counters=counters,
            min_chars_legacy=min_chars_legacy,
        )
    except Exception:
        # Record a failure and continue other files
        counters["worker.exceptions"] += 1
    return {"counters": dict(counters)}

# ===========================
# Parallel directory walk
# ===========================
def walk_and_process_parallel(
    input_dir: Path,
    output_dir: Path,
    exclude_dirs: set[str],
    keep_docstrings: bool,
    top_level_only: bool,
    max_comment_ratio: float,
    min_chars_legacy: int,
    errors_policy: str,
    tab_width: int,
    on_tok_err: str,
    min_lines: int,
    max_lines: int,
    require_docstrings: bool,
    use_ruff: bool,
    ruff_enforce: bool,
    state_db_path: Path,
    counters: Counter,
    workers: int,
    chunksize: int = 16,
) -> None:
    # Gathers .py files and process them in a multiprocessing pool. Merges per-file counters at the end.

    # 1) Collect file list and account for pruned dirs 
    file_paths: list[str] = []
    for root, dirnames, filenames in os.walk(input_dir):
        before = len(dirnames)
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs]
        counters["dirs.pruned"] += before - len(dirnames)
        root_path = Path(root)
        for fn in filenames:
            if fn.endswith(".py"):
                file_paths.append(str(root_path / fn))

    # Quick exit
    if not file_paths:
        return

    # Build a worker with static args
    worker_fn = partial(
        _worker_process_one,
        input_dir_str=str(input_dir),
        output_dir_str=str(output_dir),
        keep_docstrings=bool(keep_docstrings),
        top_level_only=bool(top_level_only),
        max_comment_ratio=float(max_comment_ratio),
        errors_policy=str(errors_policy),
        tab_width=int(tab_width),
        on_tok_err=str(on_tok_err),
        min_lines=int(min_lines),
        max_lines=int(max_lines),
        require_docstrings=bool(require_docstrings),
        use_ruff=bool(use_ruff),
        ruff_enforce=bool(ruff_enforce),
        min_chars_legacy=int(min_chars_legacy),
    )

    # Pool execution
    nprocs = (cpu_count() or 1) if workers <= 0 else max(1, workers)
    with Pool(
        processes=nprocs,
        initializer=_worker_init,
        initargs=(str(state_db_path),),
        maxtasksperchild=200,   
    ) as pool:
        for res in pool.imap_unordered(worker_fn, file_paths, chunksize):
            c = res.get("counters", {})
            for k, v in c.items():
                counters[k] += int(v)

# ---------------------------
# Walk driver + reporting
# ---------------------------

def walk_and_process(
    input_dir: Path,
    output_dir: Path,
    exclude_dirs: set[str],
    keep_docstrings: bool,
    top_level_only: bool,
    max_comment_ratio: float,
    min_chars_legacy: int,
    errors_policy: str,
    tab_width: int,
    on_tok_err: str,
    min_lines: int,
    max_lines: int,
    require_docstrings: bool,
    use_ruff: bool,
    ruff_enforce: bool,
    state: StateDB,
    counters: Counter,
) -> None:
    for root, dirnames, filenames in os.walk(input_dir):
        before = len(dirnames)
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs]
        counters["dirs.pruned"] += before - len(dirnames)
        root_path = Path(root)
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            process_file(
                root_path / fn,
                input_dir,
                output_dir,
                keep_docstrings,
                top_level_only,
                max_comment_ratio,
                errors_policy,
                tab_width,
                on_tok_err,
                min_lines,
                max_lines,
                require_docstrings,
                use_ruff,
                ruff_enforce,
                state,
                counters,
                min_chars_legacy,
            )


def print_summary(counters: Counter) -> None:
    print("\n=== INCREMENTAL QUALITY RUN SUMMARY ===")
    for key in sorted(counters):
        print(f"{key:34s}: {counters[key]}")
    print("=======================================\n")


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quality-first cleaner with persistent, cross-run dedupe and file skip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="Data/python_files", help="Input directory")
    p.add_argument("--output", default="cleaned_codebase_quality", help="Directory to write chunked files")
    p.add_argument("--exclude-dir", action="append", default=["test", ".github", "__pycache__", "examples", "docs"], help="Directory names to exclude")

    # For quality
    p.add_argument("--keep-docstrings", action="store_true", default=True, help="Keep docstrings (default: True)")
    p.add_argument("--no-keep-docstrings", dest="keep_docstrings", action="store_false", help="Disable keep-docstrings")
    p.add_argument("--top-level-only", action="store_true", default=True, help="Only split on top-level defs (default: True)")
    p.add_argument("--allow-nested", dest="top_level_only", action="store_false", help="Also split nested defs")
    p.add_argument("--max-comment-ratio", type=float, default=-1.0, help="Skip files whose removed-char ratio exceeds this; set <0 to disable")
    p.add_argument("--min-chars", type=int, default=0, help="Legacy minimal non-whitespace chars per chunk (additional to line gates)")
    p.add_argument("--errors", type=str, choices=["strict", "ignore", "replace"], default="strict", help="How to handle decoding errors while reading files")
    p.add_argument("--tab-width", type=int, default=4, help="Tab width used when normalizing leading indentation before tokenizing")
    p.add_argument("--on-tokenize-error", choices=["skip", "fallback", "keep"], default="skip", help="Behavior if tokenization fails (default: skip for quality mode)")
    p.add_argument("--min-lines", type=int, default=5, help="Minimum lines per emitted chunk")
    p.add_argument("--max-lines", type=int, default=1200, help="Maximum lines per emitted chunk (0 to disable)")
    p.add_argument("--require-docstrings", action="store_true", default=False, help="Require a docstring on each function/class")
    p.add_argument("--use-ruff", action="store_true", default=False, help="Run ruff check (E,F) on chunks if available")
    p.add_argument("--ruff-enforce", action="store_true", default=False, help="Skip chunks that fail ruff (requires --use-ruff)")

    # state & dedupe
    p.add_argument("--state-db", type=str, default="cleaned_codebase_quality/state.sqlite", help="SQLite file to store persistent state")
    p.add_argument("--seed-manifest", action="append", default=[], help="CSV manifest(s) from prior runs to seed chunk hashes")
    p.add_argument("--scan-output", action="store_true", default=False, help="Scan current output dir to seed chunk hashes on startup")

    # reporting
    p.add_argument("--report-json", type=str, default=None, help="Write audit counters to this JSON file")
    p.add_argument("--report-csv", type=str, default=None, help="Write audit counters to this CSV file")

    # Parallelism Control
    p.add_argument("--workers", type=int, default=0, help="Worker processes; 0=auto (all cores)")


    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    exclude = {d.lower() for d in (args.exclude_dir or [])}
    report_json = Path(args.report_json).resolve() if args.report_json else None
    report_csv = Path(args.report_csv).resolve() if args.report_csv else None

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # State DB 
    state = StateDB(Path(args.state_db).resolve())

    # Seed known chunk hashes
    imported = 0
    if args.seed_manifest:
        mps = [Path(p).resolve() for p in args.seed_manifest]
        imported += state.preload_hashes_from_manifest(mps)
    if args.scan_output:
        imported += state.preload_hashes_by_scanning_output(output_dir)
    if imported:
        print(f"[ok] seeded {imported} known chunk hashes into state DB")

    counters: Counter = Counter()

    if int(args.workers) != 1:
        walk_and_process_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            exclude_dirs=exclude,
            keep_docstrings=bool(args.keep_docstrings),
            top_level_only=bool(args.top_level_only),
            max_comment_ratio=float(args.max_comment_ratio),
            min_chars_legacy=int(args.min_chars),
            errors_policy=str(args.errors),
            tab_width=int(args.tab_width),
            on_tok_err=str(args.on_tokenize_error),
            min_lines=int(args.min_lines),
            max_lines=int(args.max_lines),
            require_docstrings=bool(args.require_docstrings),
            use_ruff=bool(args.use_ruff),
            ruff_enforce=bool(args.ruff_enforce),
            state_db_path=Path(args.state_db).resolve(),
            counters=counters,
            workers=int(args.workers),
            chunksize=16,
        )
    else:
        walk_and_process(
            input_dir=input_dir,
            output_dir=output_dir,
            exclude_dirs=exclude,
            keep_docstrings=bool(args.keep_docstrings),
            top_level_only=bool(args.top_level_only),
            max_comment_ratio=float(args.max_comment_ratio),
            min_chars_legacy=int(args.min_chars),
            errors_policy=str(args.errors),
            tab_width=int(args.tab_width),
            on_tok_err=str(args.on_tokenize_error),
            min_lines=int(args.min_lines),
            max_lines=int(args.max_lines),
            require_docstrings=bool(args.require_docstrings),
            use_ruff=bool(args.use_ruff),
            ruff_enforce=bool(args.ruff_enforce),
            state=state,
            counters=counters,
        )

    print_summary(counters)

    if report_json:
        with report_json.open("w", encoding="utf-8") as f:
            json.dump({"counters": dict(counters)}, f, indent=2)
        print(f"[ok] wrote audit JSON: {report_json}")
    if report_csv:
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric","count"])
            for key in sorted(counters):
                w.writerow([key, counters[key]])
        print(f"[ok] wrote audit CSV: {report_csv}")

    state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
