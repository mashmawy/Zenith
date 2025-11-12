"""
prepare_tinystories.py
Download and prepare TinyStories dataset for training
"""

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors 
import os
import json


def download_tinystories():
    """
    Download TinyStories dataset from Hugging Face
    """
    print("Downloading TinyStories dataset...")
    print("This may take a few minutes on first run...")
    
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    
    print(f"\n✅ Dataset downloaded!")
    print(f"Train examples: {len(dataset['train']):,}")
    print(f"Validation examples: {len(dataset['validation']):,}")
    
    # Show sample
    print("\n📖 Sample story:")
    print("="*60)
    print(dataset['train'][0]['text'][:500] + "...")
    print("="*60)
    
    return dataset


def train_tokenizer_on_tinystories(dataset, vocab_size=8000, save_path="./tinystories_tokenizer"):
    """
    Train a BPE tokenizer specifically on TinyStories
    
    Uses smaller vocab since stories use simple language
    """
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Configure
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
    )
    
    # Text iterator
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset['train']), batch_size):
            batch = dataset['train'][i:i+batch_size]
            yield batch['text']
    
    # Train
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(f"{save_path}/tokenizer.json")
    
    print(f"✅ Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_length=512, save_path="./tinystories_processed"):
    """
    Tokenize all stories and prepare for training
    
    max_length=512 is good for TinyStories (stories are ~200-300 words)
    """
    print(f"\nTokenizing dataset (max_length={max_length})...")
    
    def tokenize_function(examples):
        # Tokenize
        tokens = tokenizer.encode_batch(examples['text'])
        
        # Convert to IDs and create chunks
        all_input_ids = []
        all_attention_masks = []
        
        for token in tokens:
            ids = token.ids
            
            # Truncate or pad to max_length
            if len(ids) > max_length:
                ids = ids[:max_length]
            
            # Create attention mask
            attention_mask = [1] * len(ids)
            
            # Pad if needed
            if len(ids) < max_length:
                padding_length = max_length - len(ids)
                ids = ids + [0] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            all_input_ids.append(ids)
            all_attention_masks.append(attention_mask)
        
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks
        }
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        desc="Tokenizing stories"
    )
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    tokenized_dataset.save_to_disk(save_path)
    
    # Save metadata
    metadata = {
        'dataset': 'roneneldan/TinyStories',
        'vocab_size': tokenizer.get_vocab_size(),
        'max_length': max_length,
        'train_examples': len(tokenized_dataset['train']),
        'val_examples': len(tokenized_dataset['validation']),
    }
    
    with open(f"{save_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Tokenized dataset saved to {save_path}")
    print(f"Train examples: {len(tokenized_dataset['train']):,}")
    print(f"Validation examples: {len(tokenized_dataset['validation']):,}")
    
    return tokenized_dataset


def main():
    """
    Complete pipeline: Download → Train Tokenizer → Tokenize → Save
    """
    print("="*60)
    print("TinyStories Data Preparation Pipeline")
    print("="*60)
    
    # Step 1: Download dataset
    dataset = download_tinystories()
    
    # Step 2: Train tokenizer
    # Using smaller vocab_size=8000 since TinyStories has simple vocabulary
    tokenizer = train_tokenizer_on_tinystories(
        dataset, 
        vocab_size=8000,
        save_path="./tinystories_tokenizer"
    )
    
    # Step 3: Tokenize and save
    # Using max_length=512 since stories are short
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        max_length=512,
        save_path="./tinystories_processed"
    )
    
    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python train_tinystories.py")
    print("2. Monitor training with WandB")
    print("3. Generate stories with trained model")


if __name__ == "__main__":
    main()