#!/usr/bin/env python3
"""
AI Codebase Builder
Processes cleaned Python code chunks into an AI codebase with embeddings, categories, and LM Studio integration.
"""

import json
import sqlite3
import hashlib
import ast
import pickle
import random
import sys
import os
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Any

# Required imports without fallbacks
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class AICodebaseBuilder:
    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 1000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        # Create directory structure
        self.dirs = {
            'chunks': self.output_dir / 'chunks',
            'embeddings': self.output_dir / 'embeddings',
            'indexes': self.output_dir / 'indexes',
            'datasets': self.output_dir / 'datasets',
            'documentation': self.output_dir / 'documentation',
            'categories': self.output_dir / 'categories',
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = Counter()
        
        # Code categories based on imports and patterns
        self.category_rules = {
            'data_science': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly'],
            'machine_learning': ['tensorflow', 'torch', 'keras', 'sklearn', 'xgboost', 'lightgbm'],
            'web_development': ['flask', 'django', 'fastapi', 'requests', 'aiohttp', 'websocket'],
            'async_programming': ['asyncio', 'aiohttp', 'aiofiles', 'trio', 'anyio'],
            'database': ['sqlalchemy', 'pymongo', 'redis', 'psycopg2', 'sqlite3'],
            'testing': ['pytest', 'unittest', 'mock', 'nose', 'hypothesis'],
            'automation': ['selenium', 'pyautogui', 'schedule', 'fabric', 'paramiko'],
            'gui': ['tkinter', 'PyQt5', 'PyQt6', 'kivy', 'pygame', 'pyqt'],
            'scientific': ['sympy', 'astropy', 'biopython', 'rdkit', 'qiskit'],
            'cloud': ['boto3', 'azure', 'google.cloud', 'kubernetes', 'docker'],
        }

    def find_python_files(self) -> List[ Path ]:
        """Find all Python files in the input directory."""
        print(f"\nSearching for Python files in: {self.input_dir}")
        
        # Try multiple patterns to ensure we find files
        patterns = ['*.py', '**/*.py']
        py_files = set()
        
        for pattern in patterns:
            files = list(self.input_dir.glob(pattern))
            py_files.update(files)
            print(f"Pattern '{pattern}' found: {len(files)} files")
        
        # Also try rglob for recursive search
        if len(py_files) == 0:
            print("Trying recursive search...")
            py_files = set(self.input_dir.rglob('*.py'))
        
        py_files = sorted(list(py_files))
        print(f" Total unique Python files found: {len(py_files)}")
        
        if len(py_files) == 0:
            print(f"\n No Python files found in {self.input_dir}")
            print(f" Please check that the directory contains .py files")
            print(f" Directory exists: {self.input_dir.exists()}")
            if self.input_dir.exists():
                # Show what's in the directory
                items = list(self.input_dir.iterdir())[:10]
                print(f" First few items in directory: {items}")
        
        return py_files

    def process_file_to_jsonl(self, py_file: Path) -> Optional[Dict]:
        # Process a single Python file to JSONL entry. 
        try:
            code = py_file.read_text(encoding='utf-8', errors='ignore')
            
            if not code.strip():
                self.stats['empty_files'] += 1
                return None
            
            # Extract metadata from filename
            # Expected format: repo_module_type_name_hash.py
            parts = py_file.stem.split('_')
            
            # Flexible parsing for different filename formats
            if len(parts) >= 3:
                chunk_type = parts[-3] if parts[-3] in ['function', 'class', 'async_function'] else 'unknown'
                chunk_name = parts[-2] if len(parts) > 2 else 'unknown'
            else:
                chunk_type = 'unknown'
                chunk_name = py_file.stem
            
            # Create entry
            entry = {
                'id': hashlib.sha256(code.encode()).hexdigest()[:16],
                'type': chunk_type,
                'name': chunk_name,
                'code': code,
                'tokens': len(code.split()),
                'lines': len(code.splitlines()),
                'source_file': py_file.name,
                'full_path': str(py_file),
                'category': 'uncategorized' 
            }
            
            self.stats['processed_files'] += 1
            return entry
            
        except Exception as e:
            self.stats['error_files'] += 1
            print(f" Error processing {py_file}: {e}")
            return None

    def categorize_code(self, entry: Dict) -> str:
        """Categorize code based on imports and patterns."""
        code = entry['code']
        
        try:
            tree = ast.parse(code)
            
            # Extract all imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            # Check categories
            category_scores = Counter()
            for category, keywords in self.category_rules.items():
                for keyword in keywords:
                    if keyword in imports:
                        category_scores[category] += 1
            
            # Return the best matching category
            if category_scores:
                return category_scores.most_common(1)[0][0]
            
            # Check for specific patterns in code
            code_lower = code.lower()
            if 'test_' in code_lower or 'assert' in code_lower:
                return 'testing'
            elif 'async def' in code:
                return 'async_programming'
            elif 'class ' in code and ('__init__' in code or 'self.' in code):
                return 'object_oriented'
            
        except:
            pass
        
        return 'general'

    def create_datasets(self):
        """Phase 2 & 3: Create JSONL datasets and categorize."""
        print("\n Phase 2: Creating datasets...")
        
        py_files = self.find_python_files()
        if not py_files:
            return False
        
        all_entries = []
        categories = defaultdict(list)
        
        # Process files in batches
        main_dataset = self.dirs['datasets'] / 'code_chunks.jsonl'
        
        with open(main_dataset, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(py_files), self.batch_size), desc="Processing files"):
                batch = py_files[i:i + self.batch_size]
                
                for py_file in batch:
                    entry = self.process_file_to_jsonl(py_file)
                    if entry:
                        # Categorize
                        entry['category'] = self.categorize_code(entry)
                        categories[entry['category']].append(entry)
                        
                        # Write to main dataset
                        f.write(json.dumps(entry) + '\n')
                        all_entries.append(entry)
        
        print(f" Created main dataset with {len(all_entries)} entries")
        
        # Write category-specific datasets
        print("\n Creating category-specific datasets...")
        for category, entries in categories.items():
            cat_file = self.dirs['categories'] / f'{category}.jsonl'
            with open(cat_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
            print(f"  {category}: {len(entries)} entries")
        
        # Save statistics
        self.save_statistics(all_entries, categories)
        
        return True

    def create_embeddings(self, max_chunks: int = 10000):
        """Phase 4: Create embeddings and search index."""
        print("\n Phase 4: Creating embeddings...")
        
        # Load data
        dataset_file = self.dirs['datasets'] / 'code_chunks.jsonl'
        if not dataset_file.exists():
            print(" Dataset file not found. Run dataset creation first.")
            return
        
        codes = []
        metadata = []
        
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_chunks:
                    print(f" Limiting to {max_chunks} chunks for embedding")
                    break
                    
                entry = json.loads(line)
                # Truncate code for embedding (most models have token limits)
                code_snippet = entry['code'][:1000]
                codes.append(f"{entry['type']} {entry['name']}: {code_snippet}")
                metadata.append({
                    'id': entry['id'],
                    'type': entry['type'],
                    'name': entry['name'],
                    'category': entry['category']
                })
        
        if not codes:
            print(" No code chunks to embed")
            return
        
        print(f" Loaded {len(codes)} code chunks")
        
        # Initialize model
        print(" Loading embedding model...")
        try:
            # Use a smaller model if processing many chunks
            if len(codes) > 5000:
                model = SentenceTransformer('all-MiniLM-L6-v2')  
            else:
                model = SentenceTransformer('microsoft/codebert-base')  
        except:
            model = SentenceTransformer('all-MiniLM-L6-v2')  
        
        # Generate embeddings in batches
        print(f" Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(codes), batch_size), desc="Embedding batches"):
            batch = codes[i:i + batch_size]
            embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        print(" Creating FAISS index...")
        dimension = all_embeddings.shape[1]
        
        # Use a simple index for smaller datasets, IVF for larger
        if len(codes) < 10000:
            index = faiss.IndexFlatL2(dimension)
        else:
            nlist = min(100, len(codes) // 100)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(all_embeddings.astype('float32'))
        
        index.add(all_embeddings.astype('float32'))
        
        # Save index and metadata
        index_path = self.dirs['indexes'] / 'code_index.faiss'
        metadata_path = self.dirs['indexes'] / 'metadata.pkl'
        
        faiss.write_index(index, str(index_path))
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save the model name for later use
        with open(self.dirs['indexes'] / 'model_info.json', 'w') as f:
            json.dump({'model_name': model.get_sentence_embedding_dimension()}, f)
        
        print(f" Index created with {index.ntotal} vectors")

    def create_instruction_datasets(self):
        """Phase 5: Create instruction-following datasets for fine-tuning."""
        print("\n Phase 5: Creating instruction datasets...")
        
        dataset_file = self.dirs['datasets'] / 'code_chunks.jsonl'
        if not dataset_file.exists():
            print(" Dataset file not found")
            return
        
        instruction_templates = [
            "Write a Python {type} named '{name}' that {action}",
            "Create a {type} called '{name}' in Python",
            "Implement a Python {type} with the name '{name}'",
            "Generate Python code for a {type} named '{name}'",
            "Provide a Python implementation of {type} '{name}'",
        ]
        
        actions = [
            "follows best practices",
            "is well-documented",
            "is efficient and clean",
            "handles edge cases",
            "is production-ready"
        ]
        
        # Create different instruction formats
        formats = {
            'alpaca': self.dirs['datasets'] / 'alpaca_format.jsonl',
            'chat': self.dirs['datasets'] / 'chat_format.jsonl',
            'completion': self.dirs['datasets'] / 'completion_format.jsonl'
        }
        
        entries_processed = 0
        
        with open(dataset_file, 'r') as f:
            # Open all output files
            writers = {name: open(path, 'w') for name, path in formats.items()}
            
            for line in tqdm(f, desc="Creating instruction datasets"):
                entry = json.loads(line)

                if entry['tokens'] < 50 or entry['tokens'] > 2000:
                    continue
                
                template = random.choice(instruction_templates)
                action = random.choice(actions)
                
                instruction = template.format(
                    type=entry['type'],
                    name=entry['name'],
                    action= action
                )
                
                # Alpaca format
                alpaca_entry = {
                    'instruction': instruction,
                    'input': '',
                    'output': entry['code']
                }
                writers['alpaca'].write(json.dumps(alpaca_entry) + '\n')
                
                # Chat format
                chat_entry = {
                    'messages': [
                        {'role': 'user', 'content': instruction},
                        {'role': 'assistant', 'content': entry['code']}
                    ]
                }
                writers['chat'].write(json.dumps(chat_entry) + '\n')
                
                # Completion format
                completion_entry = {
                    'prompt': f"### Instruction:\n{instruction}\n\n### Response:\n",
                    'completion': entry['code']
                }
                writers['completion'].write(json.dumps(completion_entry) + '\n')
                
                entries_processed += 1
            
            # Close all files
            for writer in writers.values():
                writer.close()
        
        print(f" Created {entries_processed} instruction entries")

    def create_lm_studio_config(self):
        """Phase 6: Create LM Studio configuration and integration files."""
        print("\n Phase 6: Creating LM Studio configuration...")
        
        # Configuration for LM Studio
        config = {
            "codebase_info": {
                "name": "Python AI Codebase",
                "version": "1.0",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_chunks": self.stats['processed_files'],
                "categories": list(self.category_rules.keys())
            },
            "model_settings": {
                "context_length": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            },
            "rag_settings": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k_retrieval": 5,
                "similarity_threshold": 0.7
            },
            "prompt_templates": {
                "code_generation": "### Context:\n{context}\n\n### Task:\n{task}\n\n### Code:\n",
                "code_explanation": "### Code:\n{code}\n\n### Explanation:\n",
                "code_improvement": "### Original Code:\n{code}\n\n### Improved Version:\n",
                "code_completion": "### Partial Code:\n{partial}\n\n### Complete Code:\n"
            },
            "training_config": {
                "batch_size": 4,
                "learning_rate": 5e-5,
                "epochs": 3,
                "warmup_steps": 100,
                "dataset_format": "alpaca"
            }
        }
        
        config_file = self.output_dir / 'lm_studio_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create a sample prompt library
        prompts_file = self.output_dir / 'prompt_library.json'
        prompts = {
            "examples": [
                {
                    "name": "Generate FastAPI endpoint",
                    "prompt": "Create a FastAPI endpoint that handles user authentication with JWT tokens"
                },
                {
                    "name": "Data processing pipeline",
                    "prompt": "Write a data processing pipeline using pandas that cleans and transforms CSV data"
                },
                {
                    "name": "Async web scraper",
                    "prompt": "Implement an async web scraper using aiohttp and BeautifulSoup"
                }
            ]
        }
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        # Create integration script
        integration_script = self.output_dir / 'lm_studio_integration.py'
        with open(integration_script, 'w') as f:
            f.write(self.get_integration_script())
        
        print(f" Created LM Studio configuration files")

    def get_integration_script(self) -> str:
        """Generate the LM Studio integration script."""
        return '''#!/usr/bin/env python3
"""
LM Studio Integration Script
Use this to query your AI codebase and prepare contexts for LM Studio.
"""

import json
import pickle
from pathlib import Path

class CodebaseQuery:
    def __init__(self, codebase_dir):
        self.codebase_dir = Path(codebase_dir)
        self.load_metadata()
    
    def load_metadata(self):
        # Load the metadata
        metadata_file = self.codebase_dir / 'indexes' / 'metadata.pkl'
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []
    
    def search_by_category(self, category):
        """Search for code in a specific category."""
        category_file = self.codebase_dir / 'categories' / f'{category}.jsonl'
        if not category_file.exists():
            return []
        
        results = []
        with open(category_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    
    def get_random_examples(self, n=5):
        """Get random code examples for testing."""
        import random
        dataset_file = self.codebase_dir / 'datasets' / 'code_chunks.jsonl'
        
        examples = []
        with open(dataset_file, 'r') as f:
            lines = f.readlines()
            for _ in range(min(n, len(lines))):
                line = random.choice(lines)
                examples.append(json.loads(line))
        
        return examples

if __name__ == "__main__":
    # Example usage
    query = CodebaseQuery('.')
    examples = query.get_random_examples(3)
    
    for ex in examples:
        print(f"Type: {ex['type']}, Name: {ex['name']}")
        print(f"Category: {ex['category']}")
        print(f"Lines: {ex['lines']}, Tokens: {ex['tokens']}")
        print("-" * 50)
'''

    def save_statistics(self, all_entries: List[Dict], categories: Dict):
        """Save detailed statistics about the codebase."""
        stats = {
            'total_files': self.stats['processed_files'],
            'total_entries': len(all_entries),
            'empty_files': self.stats['empty_files'],
            'error_files': self.stats['error_files'],
            'categories': {cat: len(entries) for cat, entries in categories.items()},
            'avg_tokens': sum(e['tokens'] for e in all_entries) / len(all_entries) if all_entries else 0,
            'avg_lines': sum(e['lines'] for e in all_entries) / len(all_entries) if all_entries else 0,
            'total_tokens': sum(e['tokens'] for e in all_entries),
            'total_lines': sum(e['lines'] for e in all_entries)
        }
        
        stats_file = self.output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create a readable summary
        summary_file = self.output_dir / 'README.md'
        with open(summary_file, 'w') as f:
            f.write("# AI Codebase Summary\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Statistics\n\n")
            f.write(f"- **Total Files Processed**: {stats['total_files']:,}\n")
            f.write(f"- **Total Code Chunks**: {stats['total_entries']:,}\n")
            f.write(f"- **Total Tokens**: {stats['total_tokens']:,}\n")
            f.write(f"- **Total Lines**: {stats['total_lines']:,}\n")
            f.write(f"- **Average Tokens/Chunk**: {stats['avg_tokens']:.1f}\n")
            f.write(f"- **Average Lines/Chunk**: {stats['avg_lines']:.1f}\n\n")
            f.write("## Categories\n\n")
            for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{cat}**: {count:,} chunks\n")
            f.write("\n## Directory Structure\n\n")
            f.write("```\n")
            f.write("ai_codebase/\n")
            f.write("├── datasets/          # Main datasets in various formats\n")
            f.write("├── categories/        # Category-specific datasets\n")
            f.write("├── indexes/          # Search indexes and embeddings\n")
            f.write("├── documentation/    # Documentation and config\n")
            f.write("├── lm_studio_config.json\n")
            f.write("├── statistics.json\n")
            f.write("└── README.md\n")
            f.write("```\n")

    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("=" * 60)
        print(" AI CODEBASE BUILDER - STARTING FULL PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 2 & 3: Create datasets and categorize
        if not self.create_datasets():
            print("\n Failed to create datasets. Exiting.")
            return False
        
        # Phase 4: Create embeddings 
        self.create_embeddings(max_chunks=10000)  # Limit for performance
        
        # Phase 5: Create instruction datasets
        self.create_instruction_datasets()
        
        # Phase 6: Create LM Studio configuration
        self.create_lm_studio_config()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f" PIPELINE COMPLETE in {elapsed:.1f} seconds")
        print(f" Output directory: {self.output_dir}")
        print("=" * 60)
        
        # Print final statistics
        print("\n Final Statistics:")
        stats_file = self.output_dir / 'statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                print(f"  Total chunks: {stats['total_entries']:,}")
                print(f"  Total tokens: {stats['total_tokens']:,}")
                print(f"  Categories: {len(stats['categories'])}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Build AI-ready codebase from cleaned Python files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        default='Data/cleaned_codebase',
        help='Input directory with cleaned Python files'
    )
    parser.add_argument(
        '--output',
        default='ai_codebase',
        help='Output directory for AI codebase'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for processing files'
    )
    parser.add_argument(
        '--max-embeddings',
        type=int,
        default=10000,
        help='Maximum number of chunks to create embeddings for'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not Path(args.input).exists():
        print(f" Error: Input directory '{args.input}' does not exist!")
        print(f" Please run your cleaning script first or specify the correct path.")
        return 1
    
    # Create and run builder
    builder = AICodebaseBuilder(
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    success = builder.run_full_pipeline()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
