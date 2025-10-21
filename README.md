### AI Vector Knowledge Base
# AI Vector Knowledge Base is an automated pipeline that harvests Python repositories from GitHub, cleans the raw python scripts and extracts good-quality code, categorizes it, generates embeddings for vector search capabilities, and integrates seamlessly with LM Studio for retrieval augmented generation.

# Overview
This consists of three scripts that work in sequence to build a vector database from raw Python source code:

GitHub_Harvester.py - GitHub repository harvester
Cleaner.py - High-quality Python code cleaner
Create_Codebase.py - AI-ready dataset builder with embeddings
GitHub Repositories => Cleaned Python Chunks => Embeddings => FAISS Vector DB => LM Studio Integration

# Repository Structure
After running the pipeline, the output directory will contain:

ai_codebase/
├── datasets/              # JSONL files in multiple formats  
├── categories/            # Category-specific JSONL chunks 
├── indexes/               # FAISS embeddings and metadata 
├── documentation/         # Docs and config 
├── lm_studio_config.json  # LM Studio integration settings  
└── README.md    

### How To Run - Description - Args
## List-Repos-GitHub.py

# Description
Harvests Python repositories from GitHub using their API with time window, rate limiting handling, and adaptive crawling.

# Features
- Adaptive search based on push dates
- Parallel downloading of repository contents
- Configurable filtering by stars, file types
- Persistent state tracking
- Dashboard visualization
- Retry logic for network errors

# CLI Arguments
usage: List-Repos-GitHub.py [-h] [--github-token GITHUB_TOKEN] [--min-stars MIN_STARS]
                            [--language LANGUAGE] [--per-page PER_PAGE]
                            [--max-pages MAX_PAGES] [--start-days-ago START_DAYS_AGO]
                            [--window-min-seconds WINDOW_MIN_SECONDS]
                            [-- window-overlap_seconds WINDOW_OVERLAP_SECONDS]
                            [--sleep-between-cycles SLEEP_CYCLES]
                            [-- sleep-core-api SLEEP_CORE]
                            [--download-workers DOWNLOAD_WORKERS]
                            [-- max-files-per_repo MAX_FILES_PER_REPO]
                            [--repos-dir REPOS_DIR] [--dry-run DRY_RUN]
                            [--persistent-skip-repos PERSISTENT_SKIP_REPOS]
                            [--exclude-path-regex EXCLUDE_PATH_REGEX]

 # positional arguments:
  --github_token GITHUB_TOKEN    GitHub personal access token  
  --min-stars MIN_STARS          Minimum stars filter 
  --language LANGUAGE            Language filter 
  -- per_page PER_PAGE           Results per page 
  -- max_pages MAX_PAGES         Maximum pages 
  --start_days_ago START_DAYS_AGO Days ago 
  --window_min_seconds WINDOW_MIN_SECONDS Min window seconds 
  --window_overlap_seconds WINDOW_OVERLAP_SECONDS overlap 
  --sleep_between_cycles SLEEP_CYCLES Sleep between cycles 
  -- sleep_core_api SLEEP_CORE   Core API sleep 
  --download_workers DOWNLOAD_WORKERS Workers 
  --max_files_per_repo MAX_FILES_PER_REPO Max files 
  --repos_dir REPOS_DIR          Output directory 
  --dry-run DRY_RUN              Dry run 
  --persistent_skip_repos PERSISTENT_SKIP_REPOS Persist skip 
  --exclude_path_regex EXCLUDE_PATH_REGEX Exclude regex 


## Clean-Github-Files.py
# Description
Filters Python source files, removes comments/docstrings optionally, ensures syntax is valid, and picks clean code chunks.

# Features
- Syntax validation
- Docstring stripping
- Complexity filtering
- Duplicate removal
- Comment ratio gating
- Ruff analysis
- SQLite duplication avoidance
- Parallel processing

# CLI Arguments
usage: Clean-Github-Files.py [-h] [--input INPUT]
                            [--output OUTPUT]
                            [--exclude-dir EXCLUDE_DIR]
                            [--keep-docstrings]
                            [--top-level-only]
                            [--max-comment-ratio MAX_COMMENT_RATIO]
                            [--min-chars MIN_CHARS]
                            [--errors {strict,ignore.replace}]
                            --tab-width TAB_WIDTH
                            [--on-tokenize-error {skip fallback keep}]
                            [--min-lines MIN_LINES]
                            [--max-lines MAX_LINES]
                            [--require-docstrings]
                            [--use-ruff]
                            [--ruff-enforce]
                            [--state-db STATE_DB]
                            [-- seed-manifest SEED_MANIFEST]
                            [--scan-output SCAN_OUTPUT]
                            [--report-json REPORT_JSON]
                            [--report-csv REPORT_CSV]
                            [--workers WORKERS]

# positional arguments:
  --input INPUT                  Input directory 
  --output OUTPUT                Output directory 
  --exclude_dir EXCLUDE_DIR      Exclude directories 
  --keep-docstrings              Keep docstrings 
  --top-level-only               Top level defs 
  --max-comment_ratio MAX_COMMENT_RATIO Max comment ratio 
  --min_chars MIN_CHARS          Min chars 
  --errors {strict ignore replace} Encoding errors 
  --tab_width TAB_WIDTH          Tab width 
  --on_tokenize_error {skip fallback keep} tokenize 
  -- min-lines MIN_LINES         Minimum lines 
  --max_lines MAX_LINES          Maximum lines 
  --require-docstrings           Require docstrings 
  --use_ruff                     Use ruff 
  --ruff_enforce                 Ruff enforce 
  --state_db STATE_DB            SQLite state 
  --seed_manifest SEED_MANIFEST Seed manifests 
  --scan_output SCAN_OUTPUT      Scan output 
  --report_json REPORT_JSON      JSON audit 
  --report_csv REPORT_CSV        CSV audit 
  --workers WORKERS              Workers 


## Create_Codebase.py
# Description
Converts cleaned Python chunks into an AI vector database with embeddings, categorization.

# Features
- Generates embeddings
- FAISS vector indexing
- Code categorization
- Instruction fine-tuning datasets
- LM Studio integration
- Prompt templates
- Comprehensive statistics

# CLI Arguments
usage: Create_Codebase.py [-h] [--input INPUT] [--output OUTPUT]
                         [--batch-size BATCH_SIZE]

 # positional arguments:
 --input INPUT                  input directory 
 --output OUTPUT                output directory 
 --batch_size BATCH_SIZE        Batch size 
 --max_embeddings MAX_EMBEDDINGS Embeddings 

### Final
This project generates:
- Complete FAISS vector index
- JSON datasets
- LM Studio configuration
- Prompt templates
- Code categorization

This project is still a work in progress!

Feel free to fork the repository, submit PRs, raise issues, or suggest new features!

