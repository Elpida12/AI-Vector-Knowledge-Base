# AI Vector Knowledge Base
AI Vector Knowledge Base is an automated pipeline that harvests Python repositories from GitHub, cleans the raw python scripts and extracts good-quality code, categorizes it, generates embeddings for vector search capabilities, and integrates seamlessly with LM Studio for retrieval augmented generation.

## Overview
This consists of three scripts that work in sequence to build a vector database from raw Python source code:

* GitHub_Harvester.py - GitHub repository harvester  
* Cleaner.py - High-quality Python code cleaner  
* Create_Codebase.py - AI-ready dataset builder with embeddings  

```text
GitHub Repositories => Cleaned Python Chunks => Embeddings => FAISS Vector DB => LM Studio Integration
```

## Repository Structure
After running the pipeline, the output directory will contain:

```text
 ai_codebase/  
├── datasets/              # JSONL files in multiple formats  
├── categories/            # Category-specific JSONL chunks  
├── indexes/               # FAISS embeddings and metadata  
├── documentation/         # Docs and config  
├── lm_studio_config.json  # LM Studio integration settings  
└── README.md  
```

## How To Run - Description - Args
### GitHub_Harvester.py

### Description
Harvests Python repositories from GitHub using their API with time window, rate limiting handling, and adaptive crawling.

### Features
- Adaptive search based on push dates  
- Parallel downloading of repository contents  
- Configurable filtering by stars, file types  
- Persistent state tracking  
- Dashboard visualization  
- Retry logic for network errors  

### CLI Arguments  
```text
usage: GitHub_Harvester.py [-h]  
                          [--GITHUB_API_KEY TOKEN]  
                          [--MIN_STARS INT] [--PER_PAGE INT] [--MAX_PAGES INT]  
                          [--START_DAYS_AGO INT]  
                          [--WINDOW_MIN_SEC INT] [--WINDOW_OVERLAP_SEC INT]  
                          [--SLEEP_CYCLES FLOAT] [--SLEEP_CORE FLOAT]  
                          [--SEARCH_RPM_SOFT_MAX FLOAT]  
                          [--MAX_FILES_PER_REPO INT]  
                          [--DOWNLOAD_WORKERS INT]  
                          [--REPOS_DIR PATH] [--STATE_DIR PATH]  
                          [--DRY_RUN BOOL] [--PERSISTENT_SKIP_REPOS BOOL]  
                          [--EXCLUDE_PATH_REGEX REGEX]
```

### positional arguments   
```text
--GITHUB_API_KEY TOKEN                          GitHub personal access token.  
--MIN_STARS INT (default: 100)                  Minimum GitHub star count required for a repo to be downloaded.  
--PER_PAGE INT (default: 100)                   Repositories per page (max 100).  
--MAX_PAGES INT (default: 10)                   Maximum number of result pages per time window.  
--START_DAYS_AGO INT (default: 30)              If no prior state exists, start searching from “now − X days.”  
--WINDOW_MIN_SEC INT (default: 150)             Minimum time‑window length for each search cycle.  
--WINDOW_OVERLAP_SEC INT (default: 30)          Overlap added to window boundaries to avoid misses.  
--SLEEP_CYCLES FLOAT (default: 15.0)            Pause between full search cycles.  
--SLEEP_CORE FLOAT (default: 0.2)               Small sleep between non‑search API requests.  
--SEARCH_RPM_SOFT_MAX FLOAT                     Limit for search requests per minute.  
--MAX_FILES_PER_REPO INT (default: 1000)        Limit on how many Python files to download per repository.  
--DOWNLOAD_WORKERS INT (default: 10)            Number of parallel download workers.  
--REPOS_DIR PATH (default: Data/python_files)   Destination directory for downloaded repo files.  
--STATE_DIR PATH (default: state)               Directory for state files (search_state.json, processed_repos.txt).  
--DRY_RUN BOOL (default: false)                 Simulate downloads without writing files. Useful for checks without risking changes in your code base.  
--PERSISTENT_SKIP_REPOS BOOL (default: true)    Skip repeats across runs.  
--EXCLUDE_PATH_REGEX REGEX                      Case‑insensitive regex to exclude repo paths (e.g., '^(tests?|docs|examples?)/'). 

```


### Cleaner.py  
### Description  
Filters Python source files, removes comments/docstrings optionally, ensures syntax is valid, and picks clean code chunks.

### Features  
- Syntax validation  
- Docstring stripping  
- Complexity filtering  
- Duplicate removal  
- Comment ratio gating  
- Ruff analysis  
- SQLite duplication avoidance  
- Parallel processing  

### CLI Arguments  

```text
CLI Args
usage: Cleaner.py [-h]  
                  [--input INPUT]  
                  [--output OUTPUT]  
                  [--exclude-dir EXCLUDE_DIR]  
                  [--keep-docstrings | --no-keep-docstrings]  
                  [--top-level-only | --allow-nested]  
                  [--max-comment-ratio FLOAT]  
                  [--min-chars INT]  
                  [--errors {strict,ignore,replace}]  
                  [--tab-width INT]  
                  [--on-tokenize-error {skip,fallback,keep}]  
                  [--min-lines INT]  
                  [--max-lines INT]  
                  [--require-docstrings]  
                  [--use-ruff] [--ruff-enforce]  
                  [--state-db PATH]  
                  [--seed-manifest CSV]  
                  [--scan-output]  
                  [--report-json PATH]  
                  [--report-csv PATH]  
                  [--workers INT] 
```


### positional arguments  
```text
# Directory
--input INPUT                             Input directory (default Data/python_files)  
--output OUTPUT                           Output directory for cleaned chunks (default cleaned_codebase_quality)  
--exclude-dir EXCLUDE_DIR                 Directory name to exclude; can repeat (default: test,.github,__pycache__,examples,docs)  

# Quality controls  
--keep-docstrings                         Keep docstrings (default True)  
--no-keep-docstrings                      Disable keeping docstrings (sets keep_docstrings=False)  
--top-level-only                          Split only top-level defs (default True)  
--allow-nested                            Allow nested defs (sets top_level_only=False)  
--max-comment-ratio FLOAT                 Skip files with removed-char ratio > value; <0 disables (default -1)  
--min-chars INT                           Legacy min non-whitespace chars per chunk (adds to line gates)  
--errors {strict,ignore,replace}          How to handle read decoding errors (default strict)  
--tab-width INT                           Detab width when normalizing leading indentation (default 4)  
--on-tokenize-error {skip,fallback,keep}  Behavior if tokenization fails (default skip)  
--min-lines INT                           Min lines per emitted chunk (default 5)  
--max-lines INT                           Max lines per chunk (0=disable; default 1200)  
--require-docstrings                      Require a docstring per function/class  
--use-ruff                                Run ruff (E,F) on chunks if available  
--ruff-enforce                            Skip chunks that fail ruff (requires --use-ruff)  

# State & dedupe  
--state-db PATH                           SQLite file for persistent chunk-hash state  
--seed-manifest CSV                       Seed known hashes from prior CSV manifest(s); can repeat  
--scan-output                             Scan current OUTPUT dir to seed existing chunk hashes  

# Reporting  
--report-json PATH                        Write audit counters to JSON  
--report-csv PATH                         Write audit counters to CSV  

# Parallelism  
--workers INT                             Worker processes (0=all cores; default 0)  
```


### Create_Codebase.py  
### Description  
Converts cleaned Python chunks into an AI vector database with embeddings, categorization.

### Features  
- Generates embeddings  
- FAISS vector indexing  
- Code categorization  
- Instruction fine-tuning datasets  
- LM Studio integration  
- Prompt templates  
- Comprehensive statistics

### CLI Arguments  
```text
usage: Create_Codebase.py [-h]  
                          [--input INPUT]  
                          [--output OUTPUT]  
                          [--batch-size INT]  
                          [--max-embeddings INT]  
```

### positional arguments 
```text
 --input INPUT                  input directory  
 --output OUTPUT                output directory  
 --batch_size BATCH_SIZE        Batch size  
 --max_embeddings MAX_EMBEDDINGS Embeddings
```

### Final  
This project generates:  
- Complete FAISS vector index  
- JSON datasets  
- LM Studio configuration  
- Prompt templates  
- Code categorization

This project is still a work in progress!

Feel free to fork the repository, submit PRs, raise issues, or suggest new features!
