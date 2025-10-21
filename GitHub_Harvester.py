#!/usr/bin/env python3
"""
GitHub Python repository harvester
"""

from __future__ import annotations
import os
import re
import time
import json
import base64
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Set, Iterator, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.utils import requote_uri
import signal
import sys
import shutil

# =========================
# Configuration Management
# =========================

@dataclass
class Config:
    #Central configuration 
    
    # Authentication
    github_token: Optional[str] = field(default_factory=lambda: os.getenv("GITHUB_API_KEY") or os.getenv("GITHUB_TOKEN"))
    
    # Repository filters
    min_stars: int = field(default_factory=lambda: int(os.getenv("MIN_STARS", "100")))
    language: str = "python"
    
    # Search parameters
    per_page: int = field(default_factory=lambda: int(os.getenv("PER_PAGE", "100")))
    max_pages: int = field(default_factory=lambda: min(10, int(os.getenv("MAX_PAGES", "10"))))
    start_days_ago: int = field(default_factory=lambda: int(os.getenv("START_DAYS_AGO", "30")))
    
    # Time windows
    window_min_seconds: int = field(default_factory=lambda: int(os.getenv("WINDOW_MIN_SEC", "150")))
    window_overlap_seconds: int = field(default_factory=lambda: int(os.getenv("WINDOW_OVERLAP_SEC", "30")))
    
    # Rate limiting
    sleep_between_cycles: float = field(default_factory=lambda: float(os.getenv("SLEEP_CYCLES", "15")))
    sleep_core_api: float = field(default_factory=lambda: float(os.getenv("SLEEP_CORE", "0.2")))
    search_rpm_soft_max: Optional[float] = field(default_factory=lambda: float(os.getenv("SEARCH_RPM_SOFT_MAX")) if os.getenv("SEARCH_RPM_SOFT_MAX") else None)
    
    # Download settings
    max_files_per_repo: int = field(default_factory=lambda: int(os.getenv("MAX_FILES_PER_REPO", "1000")))
    download_workers: int = field(default_factory=lambda: int(os.getenv("DOWNLOAD_WORKERS", "10")))
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("REPOS_DIR", "Data/python_files")))
    state_dir: Path = field(default_factory=lambda: Path("state"))
    
    # Behavior flags
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "0") in ("1", "true", "True"))
    persistent_skip: bool = field(default_factory=lambda: os.getenv("PERSISTENT_SKIP_REPOS", "1") in ("1", "true", "True"))
    
    # Path filtering
    exclude_path_pattern: Optional[str] = field(default_factory=lambda: os.getenv("EXCLUDE_PATH_REGEX"))
    
    # Constants
    BASE_API_URL: str = "https://api.github.com"
    RAW_CONTENT_URL: str = "https://raw.githubusercontent.com"
    
    # Timing constants
    SEARCH_SLEEP_MIN: float = 0.5
    SEARCH_SLEEP_DEFAULT: float = 2.5
    RATE_LIMIT_HEADROOM: float = 0.1
    
    def __post_init__(self):
        # Create needed directories. 
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        
    @property
    def state_file(self) -> Path:
        return self.state_dir / "search_state.json"
    
    @property
    def processed_repos_file(self) -> Path:
        return self.state_dir / "processed_repos.txt"
    
    @property
    def stop_signal_file(self) -> Path:
        return Path("STOP")

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_file: str = "harvester.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("github_harvester")

# ============================================================================
# HTTP Session Management
# ============================================================================

class APIError(Exception):
    pass

class RateLimitError(APIError):
    def __init__(self, reset_time: int):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Resets at {reset_time}")

class HTTPSessionManager:
    # Manages HTTP sessions with proper connection pooling and retry logic.
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()
        self._search_sleep: float = config.SEARCH_SLEEP_DEFAULT
        
    @property
    def session(self) -> requests.Session:
        # Initialization of session with thread safety.
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = self._create_session()
        return self._session
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        
        retries = Retry(
            total=6,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        
        pool_size = max(32, self.config.download_workers * 2)
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=pool_size,
            pool_maxsize=pool_size
        )
        
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def get_headers(self, is_raw: bool = False) -> Dict[str, str]:
        # Generate headers for the request type. 
        headers = {"User-Agent": "python-harvester/2.0"}
        
        if not is_raw:
            headers.update({
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            if self.config.github_token:
                headers["Authorization"] = f"Bearer {self.config.github_token}"
                
        return headers
    
    def calculate_search_sleep(self, rate_info: Dict[str, Any]) -> float:
        # Calculate sleep time for search API. 
        if self.config.search_rpm_soft_max:
            return max(30.0 / self.config.search_rpm_soft_max, self.config.SEARCH_SLEEP_MIN)
        
        search_bucket = rate_info.get("search", {})
        remaining = search_bucket.get("remaining")
        reset_timestamp = search_bucket.get("reset")
        
        if remaining is not None and reset_timestamp:
            current_time = int(datetime.now(timezone.utc).timestamp())
            seconds_until_reset = max(reset_timestamp - current_time, 1)
            
            # Add headroom to avoid hitting the limit
            sleep_time = (seconds_until_reset / max(remaining, 1)) * (1 + self.config.RATE_LIMIT_HEADROOM)
            return max(sleep_time, self.config.SEARCH_SLEEP_MIN)
        
        return self.config.SEARCH_SLEEP_DEFAULT
    
    def check_rate_limit(self) -> Dict[str, Any]:
        # Query and return current rate limit status. 
        url = f"{self.config.BASE_API_URL}/rate_limit"
        response = self.session.get(url, headers=self.get_headers())
        response.raise_for_status()
        return response.json().get("resources", {})
    
    def close(self):
        # Clean up session resources. 
        if self._session:
            self._session.close()
            self._session = None

# ============================================================================
# GitHub API Client
# ============================================================================

class GitHubClient:
    
    def __init__(self, config: Config, session_manager: HTTPSessionManager, logger: logging.Logger):
        self.config = config
        self.session_mgr = session_manager
        self.logger = logger
        
    def _handle_rate_limit(self, response: requests.Response) -> requests.Response:
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_time = int(response.headers.get("X-RateLimit-Reset", "0"))
            current_time = int(datetime.now(timezone.utc).timestamp())
            sleep_duration = max(reset_time - current_time, 5)
            
            self.logger.warning(f"Rate limit hit. Sleeping for {sleep_duration} seconds")
            time.sleep(sleep_duration)
            
            # Retry the request
            return self.session_mgr.session.get(
                response.url,
                headers=response.request.headers,
                params=response.request.params,
                timeout=60
            )
        return response
    
    def _api_request(self, url: str, params: Optional[Dict] = None, is_raw: bool = False) -> requests.Response:
        # Make an API request with appropriate pacing. 
        is_search = "/search/" in url
        headers = self.session_mgr.get_headers(is_raw)
        
        response = self.session_mgr.session.get(url, headers=headers, params=params, timeout=60)
        
        if not is_raw:
            response = self._handle_rate_limit(response)
            response.raise_for_status()
        
        if is_search:
            time.sleep(self.session_mgr._search_sleep)
        else:
            time.sleep(self.config.sleep_core_api)
            
        return response
    
    def search_repositories(self, query: str, page: int = 1) -> Dict[str, Any]:
        # Search for repositories matching the query. 
        params = {
            "q": query,
            "sort": "updated",
            "order": "asc",
            "per_page": self.config.per_page,
            "page": page
        }
        
        url = f"{self.config.BASE_API_URL}/search/repositories"
        response = self._api_request(url, params=params)
        return response.json()
    
    def get_repository_tree(self, owner: str, repo: str, branch: str) -> List[str]:

        python_files = []
        
        try:
            branch_url = f"{self.config.BASE_API_URL}/repos/{owner}/{repo}/branches/{branch}"
            branch_info = self._api_request(branch_url).json()
            tree_sha = branch_info["commit"]["commit"]["tree"]["sha"]
            
            # Get the tree contents
            tree_url = f"{self.config.BASE_API_URL}/repos/{owner}/{repo}/git/trees/{tree_sha}"
            tree_data = self._api_request(tree_url, params={"recursive": "1"}).json()
            
            # Filter for Python files
            exclude_pattern = None
            if self.config.exclude_path_pattern:
                exclude_pattern = re.compile(self.config.exclude_path_pattern, re.IGNORECASE)
            
            for entry in tree_data.get("tree", []):
                if entry.get("type") != "blob":
                    continue
                    
                path = entry.get("path", "")
                if not path.endswith((".py", ".pyi")):
                    continue
                    
                if exclude_pattern and exclude_pattern.search(path):
                    continue
                    
                python_files.append(path)
                
            if tree_data.get("truncated"):
                self.logger.warning(f"Tree truncated for {owner}/{repo}: got {len(python_files)} Python files")
                
        except Exception as e:
            self.logger.error(f"Error getting tree for {owner}/{repo}: {e}")
            
        return python_files
    
    def download_file(self, owner: str, repo: str, branch: str, file_path: str) -> Optional[bytes]:
        if self.config.dry_run:
            return b""
        
        encoded_path = requote_uri(file_path)
        raw_url = f"{self.config.RAW_CONTENT_URL}/{owner}/{repo}/{branch}/{encoded_path}"
        
        try:
            response = self._api_request(raw_url, is_raw=True)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            self.logger.debug(f"Raw download failed for {file_path}: {e}")
        
        # Fallback to Contents API
        try:
            contents_url = f"{self.config.BASE_API_URL}/repos/{owner}/{repo}/contents/{file_path}"
            response = self._api_request(contents_url, params={"ref": branch})
            data = response.json()
            
            if isinstance(data, dict) and data.get("encoding") == "base64":
                return base64.b64decode(data.get("content", ""))
                
        except Exception as e:
            self.logger.warning(f"Failed to download {owner}/{repo}/{file_path}: {e}")
            
        return None

# ============================================================================
# Repository Processing
# ============================================================================

class RepositoryProcessor:
    
    def __init__(self, config: Config, client: GitHubClient, logger: logging.Logger, 
                 dashboard: Optional['TerminalDashboard'] = None):  
        self.config = config
        self.client = client
        self.logger = logger
        self.stats = ProcessingStats()
        self.dashboard = dashboard  
        
    def process_repository(self, repo: Dict[str, Any], processed_ids: Set[int]) -> bool:
        repo_id = repo["id"]
        owner = repo["owner"]["login"]
        name = repo["name"]
        full_name = repo.get("full_name", f"{owner}/{name}")
        branch = repo.get("default_branch", "main")
        
        if repo_id in processed_ids:
            return False
        
        # Get list of Python files
        python_files = self.client.get_repository_tree(owner, name, branch)
        
        if not python_files:
            self.logger.info(f"{full_name}: No Python files found")
            processed_ids.add(repo_id)
            self.stats.increment_repos()
            if self.dashboard:
                self.dashboard.set_last_repo(full_name, 0, 0)
                self.dashboard.render()
            return True
        
        files_to_download = python_files[:self.config.max_files_per_repo]

        downloaded_count = self._download_files_parallel(
            owner, name, branch, files_to_download
        )
        
        self.logger.info(
            f"{full_name}: Downloaded {downloaded_count}/{len(files_to_download)} files "
            f"(found {len(python_files)} total)"
        )
        
        processed_ids.add(repo_id)
        self.stats.increment_repos()
        self.stats.increment_files(downloaded_count)
        
        if self.dashboard:
            self.dashboard.set_last_repo(full_name, downloaded_count, len(files_to_download))
            self.dashboard.render()
        
        return True
    
    def _download_files_parallel(self, owner: str, name: str, branch: str, files: List[str]) -> int:
        if self.config.dry_run:
            return len(files)
        
        repo_dir = self.config.data_dir / f"{owner}-{name}"
        downloaded = 0
        
        with ThreadPoolExecutor(max_workers=self.config.download_workers) as executor:
            futures = {
                executor.submit(
                    self._download_single_file,
                    owner, name, branch, file_path, repo_dir / file_path
                ): file_path
                for file_path in files
            }
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        downloaded += 1
                except Exception as e:
                    file_path = futures[future]
                    self.logger.debug(f"Failed to download {file_path}: {e}")
                    
        return downloaded
    
    def _download_single_file(self, owner: str, name: str, branch: str, 
                             file_path: str, output_path: Path) -> bool:
        # Download a single file.
        content = self.client.download_file(owner, name, branch, file_path)
        
        if content is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            return True
            
        return False
# ============================================================================
# Search Strategy
# ============================================================================

class AdaptiveSearchStrategy:
    
    MINIMUM_WINDOW_SECONDS = 120
    MAX_RESULTS_PER_WINDOW = 1000
    
    def __init__(self, config: Config, client: GitHubClient, logger: logging.Logger):
        self.config = config
        self.client = client
        self.logger = logger
        
    def search_time_window(self, start: datetime, end: datetime) -> Iterator[Dict[str, Any]]:
        # Search repositories in a time window with adaptive splitting.
        results = list(self._search_window_impl(start, end))
        
        # Check if we hit limits
        if len(results) >= self.MAX_RESULTS_PER_WINDOW or self._is_saturated(results):
            window_duration = (end - start).total_seconds()
            
            # Split window if it's large enough
            if window_duration > self.MINIMUM_WINDOW_SECONDS:
                mid_point = start + (end - start) / 2
                
                self.logger.info(f"Splitting saturated window: {start.isoformat()} -> {end.isoformat()}")
                
                # Recursively search both halves
                yield from self.search_time_window(start, mid_point)
                yield from self.search_time_window(mid_point, end)
                return
        
        # Yield unique results
        seen_ids = set()
        for repo in results:
            repo_id = repo["id"]
            if repo_id not in seen_ids:
                seen_ids.add(repo_id)
                yield repo
    
    def _search_window_impl(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        # Execute search for a specific window.
        query = self._build_query(start, end)
        all_results = []
        
        for page in range(1, self.config.max_pages + 1):
            try:
                response = self.client.search_repositories(query, page)
                items = response.get("items", [])
                
                if not items:
                    break
                    
                all_results.extend(items)
                
                # Stop if we got less than a full page
                if len(items) < self.config.per_page:
                    break
                    
            except Exception as e:
                self.logger.error(f"Search error on page {page}: {e}")
                break
                
        return all_results
    
    def _build_query(self, start: datetime, end: datetime) -> str:
        # Build GitHub search query.
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return (
            f"is:public fork:false "
            f"language:{self.config.language} "
            f"stars:>{self.config.min_stars} "
            f"pushed:{start_str}..{end_str}"
        )
    
    def _is_saturated(self, results: List[Dict[str, Any]]) -> bool:
        # Check if results indicate we hit a limit.
        return len(results) == self.config.max_pages * self.config.per_page

# ============================================================================
# State Management
# ============================================================================

class StateManager:
    # Manages persistent state across runs.
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def load_state(self) -> Tuple[datetime, Set[int]]:
        # Load last check time and processed repository IDs.
        last_checked = datetime.now(timezone.utc) - timedelta(days=self.config.start_days_ago)
        
        if self.config.state_file.exists():
            try:
                state_data = json.loads(self.config.state_file.read_text())
                if "last_checked" in state_data:
                    last_checked = datetime.fromisoformat(state_data["last_checked"])
                    if last_checked.tzinfo is None:
                        last_checked = last_checked.replace(tzinfo=timezone.utc)
            except Exception as e:
                self.logger.warning(f"Error loading state: {e}")
        
        # Load processed repository IDs
        processed_ids = set()
        if self.config.persistent_skip and self.config.processed_repos_file.exists():
            try:
                with open(self.config.processed_repos_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            processed_ids.add(int(line))
            except Exception as e:
                self.logger.warning(f"Error loading processed repos: {e}")
                
        return last_checked, processed_ids
    
    def save_state(self, last_checked: datetime):
        # Save the last check time.
        try:
            state_data = {"last_checked": last_checked.isoformat()}
            self.config.state_file.write_text(json.dumps(state_data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def save_processed_repo(self, repo_id: int):
        if not self.config.persistent_skip:
            return
            
        try:
            with open(self.config.processed_repos_file, "a") as f:
                f.write(f"{repo_id}\n")
        except Exception as e:
            self.logger.error(f"Error saving processed repo ID: {e}")

# ============================================================================
# Statistics Tracking
# ============================================================================

class ProcessingStats:
    
    def __init__(self):
        self.repos_processed = 0
        self.files_downloaded = 0
        self.start_time = time.monotonic()
        self._lock = threading.Lock()
        
    def increment_repos(self, count: int = 1):
        with self._lock:
            self.repos_processed += count
            
    def increment_files(self, count: int = 1):
        with self._lock:
            self.files_downloaded += count
            
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self.start_time
            return {
                "repos_processed": self.repos_processed,
                "files_downloaded": self.files_downloaded,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": self._format_duration(elapsed)
            }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        # Format duration as HH:MM:SS.
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
# ============================================================================
# Terminal Dashboard for Statistics
# ============================================================================
class TerminalDashboard:
    
    # ANSI escape codes
    CLEAR_SCREEN = "\033[2J"
    MOVE_CURSOR_HOME = "\033[H"
    CLEAR_LINE = "\033[K"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    
    def __init__(self, config: Config, stats: ProcessingStats, logger: logging.Logger):
        self.config = config
        self.stats = stats
        self.logger = logger
        self.enabled = sys.stdout.isatty() and not os.getenv("NO_DASHBOARD", "0") in ("1", "true")
        self.current_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
        self.last_repo: str = "None"
        self.last_repo_files: Tuple[int, int] = (0, 0)  # (downloaded, total)
        self._last_update = 0
        self.update_interval = 0.5
        
    def set_window(self, start: datetime, end: datetime):
        self.current_window = (start, end)
        
    def set_last_repo(self, repo_name: str, downloaded: int, total: int):
        self.last_repo = repo_name
        self.last_repo_files = (downloaded, total)
        
    def render(self, force: bool = False):
        if not self.enabled:
            return
            
        # Rate limit updates to prevent excessive CPU usage
        current_time = time.monotonic()
        if not force and (current_time - self._last_update) < self.update_interval:
            return
        self._last_update = current_time
        
        # Get terminal dimensions
        try:
            cols, rows = shutil.get_terminal_size((80, 24))
        except:
            cols, rows = 80, 24
            
        # Get current stats
        stats = self.stats.get_stats()
        
        lines = []
        
        # Header
        lines.append(f"{self.BOLD}{self.CYAN}{'=' * min(cols, 80)}{self.RESET}")
        lines.append(f"{self.BOLD}{self.GREEN}GitHub Python Harvester Dashboard{self.RESET}".center(min(cols, 80)))
        lines.append(f"{self.BOLD}{self.CYAN}{'=' * min(cols, 80)}{self.RESET}")
        lines.append("")
        
        # Configuration section
        lines.append(f"{self.BOLD}Configuration:{self.RESET}")
        lines.append(f"  • Repository: {self.config.data_dir}")
        lines.append(f"  • Min Stars: {self.config.min_stars}")
        lines.append(f"  • Workers: {self.config.download_workers}")
        lines.append(f"  • Max Files/Repo: {self.config.max_files_per_repo}")
        if self.config.exclude_path_pattern:
            lines.append(f"  • Exclude Pattern: {self.config.exclude_path_pattern}")
        lines.append("")
        
        # Statistics section
        lines.append(f"{self.BOLD}Statistics:{self.RESET}")
        lines.append(f"  • Elapsed Time: {self.YELLOW}{stats['elapsed_formatted']}{self.RESET}")
        lines.append(f"  • Repos Processed: {self.GREEN}{stats['repos_processed']:,}{self.RESET}")
        lines.append(f"  • Files Downloaded: {self.GREEN}{stats['files_downloaded']:,}{self.RESET}")
        
        # Calculate rates
        if stats['elapsed_seconds'] > 0:
            repos_per_hour = (stats['repos_processed'] / stats['elapsed_seconds']) * 3600
            files_per_hour = (stats['files_downloaded'] / stats['elapsed_seconds']) * 3600
            lines.append(f"  • Repos/Hour: {repos_per_hour:.1f}")
            lines.append(f"  • Files/Hour: {files_per_hour:.1f}")
        lines.append("")
        
        # Current activity section
        lines.append(f"{self.BOLD}Current Activity:{self.RESET}")
        if self.current_window[0] and self.current_window[1]:
            window_start, window_end = self.current_window
            duration = (window_end - window_start).total_seconds()
            lines.append(f"  • Processing Window: {duration:.0f} seconds")
            lines.append(f"    From: {window_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append(f"    To:   {window_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"  • Last Repository: {self.last_repo}")
        lines.append(f"    Files: {self.last_repo_files[0]}/{self.last_repo_files[1]}")
        lines.append("")
        
        # Footer
        lines.append(f"{self.CYAN}{'─' * min(cols, 80)}{self.RESET}")
        lines.append("Press Ctrl+C to stop gracefully")
        
        # Clear screen and print dashboard
        sys.stdout.write(self.MOVE_CURSOR_HOME)
        for i, line in enumerate(lines):
            if i < rows - 1:  # Leave one line for cursor
                sys.stdout.write(line[:cols] + self.CLEAR_LINE + "\n")
        
        # Clear any remaining lines
        for i in range(len(lines), rows - 1):
            sys.stdout.write(self.CLEAR_LINE + "\n")
            
        sys.stdout.flush()
        
    def start(self):
        # Initialize dashboard display.
        if self.enabled:
            sys.stdout.write(self.HIDE_CURSOR + self.CLEAR_SCREEN)
            sys.stdout.flush()
            
    def stop(self):
        # Clean up dashboard display.
        if self.enabled:
            sys.stdout.write(self.SHOW_CURSOR)
            sys.stdout.flush()

# ============================================================================
# Main Orchestrator
# ============================================================================

class GitHubHarvester:
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = setup_logging()
        self.running = True
        
        # Initialize components
        self.session_manager = HTTPSessionManager(self.config, self.logger)
        self.client = GitHubClient(self.config, self.session_manager, self.logger)
        
        # Create stats first, then dashboard, then processor with dashboard reference
        stats = ProcessingStats()
        self.dashboard = TerminalDashboard(self.config, stats, self.logger)
        self.processor = RepositoryProcessor(self.config, self.client, self.logger, self.dashboard)
        self.processor.stats = stats  # Use the same stats instance
        
        self.search_strategy = AdaptiveSearchStrategy(self.config, self.client, self.logger)
        self.state_manager = StateManager(self.config, self.logger)
        
    def run(self):
        # Main execution loop.
        try:
            self.dashboard.start()  # Start dashboard
            self._initialize()
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested via keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.dashboard.stop()  # Stop dashboard
            self._cleanup()
    
    def _initialize(self):

        # Check authentication
        if not self.config.github_token:
            self.logger.warning(
                "No GitHub token configured. API rate limits will be very restrictive. "
                "Set GITHUB_API_KEY or GITHUB_TOKEN environment variable."
            )
        
        # Update search sleep time based on rate limits
        try:
            rate_info = self.session_manager.check_rate_limit()
            self.session_manager._search_sleep = self.session_manager.calculate_search_sleep(rate_info)
            
            search_limit = rate_info.get("search", {})
            self.logger.info(
                f"Search rate limit: {search_limit.get('remaining')}/{search_limit.get('limit')} "
                f"(sleep: {self.session_manager._search_sleep:.2f}s between requests)"
            )
        except Exception as e:
            self.logger.warning(f"Could not check rate limits: {e}")
    
    def _main_loop(self):
        last_checked, persistent_processed = self.state_manager.load_state()
        
        while self.running:
            # Check for stop signal
            if self.config.stop_signal_file.exists():
                self.logger.info("Stop signal detected")
                break
            
            # Calculate time window for this cycle
            current_time = datetime.now(timezone.utc)
            effective_end = current_time - timedelta(seconds=self.config.window_overlap_seconds)
            
            # Ensure minimum window size
            window_duration = (effective_end - last_checked).total_seconds()
            if window_duration < self.config.window_min_seconds:
                window_start = effective_end - timedelta(seconds=self.config.window_min_seconds)
            else:
                window_start = last_checked
            
            window_end = effective_end
            
            # Update dashboard with current window
            self.dashboard.set_window(window_start, window_end)
            self.dashboard.render()
            
            self.logger.info(
                f"Processing window: {window_start.isoformat()} to {window_end.isoformat()} "
                f"({(window_end - window_start).total_seconds():.0f} seconds)"
            )
            
            # Process repositories in this window
            cycle_processed = set(persistent_processed) if self.config.persistent_skip else set()
            repos_found = 0
            
            try:
                for repo in self.search_strategy.search_time_window(window_start, window_end):
                    if not self.running:
                        break
                        
                    repos_found += 1
                    processed = self.processor.process_repository(repo, cycle_processed)
                    
                    if processed and self.config.persistent_skip:
                        self.state_manager.save_processed_repo(repo["id"])
                        persistent_processed.add(repo["id"])
                        
            except Exception as e:
                self.logger.error(f"Error during repository processing: {e}")
            
            # Update state
            last_checked = window_end
            self.state_manager.save_state(last_checked)
            
            # Log statistics
            stats = self.processor.stats.get_stats()
            self.logger.info(
                f"Cycle complete. Repos found: {repos_found}, "
                f"Total processed: {stats['repos_processed']}, "
                f"Files downloaded: {stats['files_downloaded']}, "
                f"Elapsed: {stats['elapsed_formatted']}"
            )
            
            # Final dashboard update for this cycle
            self.dashboard.render(force=True)
            
            # Sleep before next cycle
            if self.running:
                time.sleep(self.config.sleep_between_cycles)

# ============================================================================
# Entry Point
# ============================================================================

def main():
    harvester = GitHubHarvester()
    
    # Graceful shutdown
    def signal_handler(signum, frame):
        harvester.logger.info(f"Received signal {signum}")
        harvester.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the harvester
    harvester.run()

if __name__ == "__main__":
    main()