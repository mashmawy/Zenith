"""
generate_stories.py
Generate new stories with trained model
"""
import sys
import os

# Get the absolute path of the directory that contains the 'model_architecture.py' file.
# In your case, it's 'E:\Work\Projects\llm'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # This steps up from 'tinystories' to 'llm'

# Add the root directory to the system path
sys.path.append(root_dir)
import torch
from tokenizers import Tokenizer
import json

from model import LLMModel, ModelConfig

 

def load_trained_model(checkpoint_path, tokenizer_path):
    """Load trained model and tokenizer"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    config = ModelConfig(**checkpoint['model_config'])
    model = LLMModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")
    
    return model, tokenizer


def generate_story(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_length=300,
    temperature=0.8,
    top_p=0.95,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate a story from a prompt
    """
    model = model.to(device)
    
    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    # Generate
    generated_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated
            generated_ids.append(next_token.item())
            
            # Update input_ids
            input_ids = torch.tensor([generated_ids]).to(device)
            
            # Stop at end token (if you have one)
            if next_token.item() == 3:  # </s> token
                break
    
    # Decode
    story = tokenizer.decode(generated_ids)
    return story


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", default="./tinystories_tokenizer")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--num_stories", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_trained_model(args.checkpoint, args.tokenizer)
    print("✅ Model loaded\n")
    
    # Generate stories
    for i in range(args.num_stories):
        print(f"Story {i+1}:")
        print("="*60)
        story = generate_story(
            model,
            tokenizer,
            prompt=args.prompt,
            temperature=args.temperature
        )
        print(story)
        print("="*60 + "\n")


if __name__ == "__main__":
    main()