"""
Data Preparation Tool for LLM Training
Converts unstructured text corpus into tokenized training-ready datasets
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Iterator
import multiprocessing as mp
from dataclasses import dataclass 
from tqdm import tqdm

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from datasets import Dataset
    import tiktoken
except ImportError:
    print("Install required packages: pip install tokenizers datasets tiktoken")
    exit(1)


@dataclass
class DataConfig:
    """Configuration for data preparation"""
    input_dir: str
    output_dir: str
    tokenizer_type: str = "bpe"  # bpe, wordpiece, or tiktoken
    vocab_size: int = 32000
    min_frequency: int = 2
    max_length: int = 2048
    stride: int = 1024  # For overlapping chunks
    num_workers: int = 4
    file_extensions: List[str] = None
    
    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = [".txt", ".md", ".json"]


class DataPreparator:
    """Handles data collection, tokenization, and dataset creation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = None
        
    def collect_files(self) -> List[Path]:
        """Recursively collect all text files from input directory"""
        input_path = Path(self.config.input_dir)
        files = []
        
        for ext in self.config.file_extensions:
            files.extend(input_path.rglob(f"*{ext}"))
        
        print(f"Found {len(files)} files")
        return files
    
    def read_file_content(self, filepath: Path) -> str:
        """Read content from a file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    
    def text_iterator(self, files: List[Path], batch_size: int = 1000) -> Iterator[List[str]]:
        """Yield batches of text for tokenizer training"""
        batch = []
        for filepath in tqdm(files, desc="Reading files"):
            content = self.read_file_content(filepath)
            if content:
                # Split large files into chunks for memory efficiency
                lines = content.split('\n')
                for line in lines:
                    if line.strip():
                        batch.append(line)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
        
        if batch:
            yield batch
    
    def train_tokenizer(self, files: List[Path]):
        """Train or load tokenizer"""
        tokenizer_path = Path(self.config.output_dir) / "tokenizer.json"
        
        if tokenizer_path.exists():
            print(f"Loading existing tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            return
        
        print(f"Training {self.config.tokenizer_type} tokenizer...")
        
        if self.config.tokenizer_type == "tiktoken":
            # Use pre-trained tiktoken (GPT-4 tokenizer)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print("Using tiktoken cl100k_base encoding")
            return
        
        # Create tokenizer based on type
        if self.config.tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            trainer = trainers.BpeTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            )
        elif self.config.tokenizer_type == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {self.config.tokenizer_type}")
        
        # Configure pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Train on text iterator
        tokenizer.train_from_iterator(
            self.text_iterator(files),
            trainer=trainer,
        )
        
        self.tokenizer = tokenizer
        
        # Save tokenizer
        os.makedirs(self.config.output_dir, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text string"""
        if isinstance(self.tokenizer, tiktoken.Encoding):
            return self.tokenizer.encode(text)
        else:
            return self.tokenizer.encode(text).ids
    
    def create_chunks(self, token_ids: List[int]) -> List[List[int]]:
        """Split token sequence into overlapping chunks"""
        chunks = []
        max_len = self.config.max_length
        stride = self.config.stride
        
        for i in range(0, len(token_ids), stride):
            chunk = token_ids[i:i + max_len]
            if len(chunk) >= max_len // 2:  # Keep chunks at least half full
                chunks.append(chunk)
        
        return chunks
    
    def process_file(self, filepath: Path) -> List[List[int]]:
        """Process a single file into tokenized chunks"""
        content = self.read_file_content(filepath)
        if not content:
            return []
        
        token_ids = self.tokenize_text(content)
        chunks = self.create_chunks(token_ids)
        return chunks
    
    def prepare_dataset(self, files: List[Path]):
        """Process all files and create training dataset"""
        print("Tokenizing and chunking files...")
        
        all_chunks = []
        
        # Process files with multiprocessing
        with mp.Pool(self.config.num_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_file, files),
                total=len(files),
                desc="Processing files"
            ))
        
        for chunks in results:
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} training examples")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_dict({
            "input_ids": all_chunks,
            "attention_mask": [[1] * len(chunk) for chunk in all_chunks]
        })
        
        # Split into train/val
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        # Save dataset
        output_path = Path(self.config.output_dir)
        split_dataset.save_to_disk(str(output_path / "dataset"))
        
        # Save metadata
        metadata = {
            "num_examples": len(all_chunks),
            "train_examples": len(split_dataset["train"]),
            "val_examples": len(split_dataset["test"]),
            "max_length": self.config.max_length,
            "vocab_size": self.config.vocab_size if not isinstance(self.tokenizer, tiktoken.Encoding) else self.tokenizer.n_vocab,
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to {output_path / 'dataset'}")
        print(f"Train examples: {metadata['train_examples']}")
        print(f"Validation examples: {metadata['val_examples']}")
    
    def run(self):
        """Execute the full data preparation pipeline"""
        print("=== LLM Data Preparation ===\n")
        
        # Collect files
        files = self.collect_files()
        if not files:
            print("No files found!")
            return
        
        # Train/load tokenizer
        self.train_tokenizer(files)
        
        # Prepare dataset
        self.prepare_dataset(files)
        
        print("\n=== Preparation Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Prepare text data for LLM training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe", "wordpiece", "tiktoken"])
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    config = DataConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        stride=args.stride,
        num_workers=args.num_workers,
    )
    
    preparator = DataPreparator(config)
    preparator.run()


if __name__ == "__main__":
    main()
