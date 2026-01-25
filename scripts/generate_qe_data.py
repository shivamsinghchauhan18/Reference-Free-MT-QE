import os
import random
import json
import argparse
import logging
from typing import List, Tuple
import torch
from comet import download_model, load_from_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoiseInjector:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _delete_ops(self, words: List[str], n: int) -> List[str]:
        """Perform n deletions."""
        if not words: return words
        temp = list(words)
        for _ in range(n):
            if not temp: break
            idx = random.randint(0, len(temp) - 1)
            temp.pop(idx)
        return temp

    def _swap_ops(self, words: List[str], n: int) -> List[str]:
        """Perform n swaps."""
        if len(words) < 2: return words
        temp = list(words)
        for _ in range(n):
            if len(temp) < 2: break
            idx = random.randint(0, len(temp) - 2)
            temp[idx], temp[idx+1] = temp[idx+1], temp[idx]
        return temp

    def _mask_span(self, words: List[str], ratio: float) -> List[str]:
        """Mask a continuous span of tokens."""
        if not words: return words
        span_len = max(1, int(len(words) * ratio))
        start_idx = random.randint(0, max(0, len(words) - span_len))
        # We replace with a generic mask token or just remove them. 
        # For MT hallucination simulation, removing or replacing with random words is better.
        # Let's simple remove them to simulate 'skipping' content.
        return words[:start_idx] + words[start_idx + span_len:]

    def apply_noise(self, text: str, level: str) -> str:
        words = text.split()
        if not words: return text
        
        # Original: Return as is
        if level == "original":
            return text

        # Light: 1 operation (swap or delete)
        if level == "light":
            op = random.choice(["delete", "swap"])
            if op == "delete":
                return " ".join(self._delete_ops(words, 1))
            else:
                return " ".join(self._swap_ops(words, 1))

        # Medium: 2-3 operations
        if level == "medium":
            num_ops = random.randint(2, 3)
            # Mix of swaps and deletes
            temp_words = list(words)
            for _ in range(num_ops):
                op = random.choice(["delete", "swap"])
                if op == "delete":
                    temp_words = self._delete_ops(temp_words, 1)
                else:
                    temp_words = self._swap_ops(temp_words, 1)
            return " ".join(temp_words)

        # Heavy: Span deletion (30-50%) or massive shuffle
        if level == "heavy":
            ratio = random.uniform(0.3, 0.5)
            return " ".join(self._mask_span(words, ratio))
            
        return text

def load_lines(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic QE Data for High Sorbian (Multi-Level Noise)")
    parser.add_argument("--src", default="MT/train.de-hsb.de", help="Path to source file")
    parser.add_argument("--ref", default="MT/train.de-hsb.hsb", help="Path to reference file")
    parser.add_argument("--output", default="data/qe_finetune_hsb.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of source lines to process")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for COMET scoring")
    
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data from {args.src} and {args.ref}...")
    src_lines = load_lines(args.src)
    ref_lines = load_lines(args.ref)
    
    if len(src_lines) != len(ref_lines):
        raise ValueError("Source and reference files must have the same number of lines.")
    
    if args.limit:
        src_lines = src_lines[:args.limit]
        ref_lines = ref_lines[:args.limit]
        
    logger.info(f"Loaded {len(src_lines)} pairs.")

    # 2. Initialize Model
    logger.info("Loading COMET-DA model...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()

    # 3. Generate Multi-Level Variations
    noise_injector = NoiseInjector()
    comet_input_data = [] # List of {"src": s, "mt": m, "ref": r}
    metadata = [] # List of {"src": s, "mt": m, "level": l}
    
    logger.info("Generating noisy hypotheses (Original, Light, Medium, Heavy)...")
    
    levels = ["original", "light", "medium", "heavy"]
    
    for i, (src, ref) in enumerate(zip(src_lines, ref_lines)):
        for level in levels:
            noisy_mt = noise_injector.apply_noise(ref, level)
            
            # Avoid duplicate empty strings if possible, but COMET works either way
            # Also avoid duplicates if noise didn't change anything (e.g. short sentence)
            
            comet_input_data.append({"src": src, "mt": noisy_mt, "ref": ref})
            metadata.append({"src": src, "mt": noisy_mt, "level": level})

    logger.info(f"Generated {len(comet_input_data)} total examples. Scoring with COMET...")
    
    # 4. Score
    model_output = model.predict(comet_input_data, batch_size=args.batch_size, gpus=1 if torch.cuda.is_available() else 0, num_workers=0)
    scores = model_output.scores

    # 5. Save
    logger.info(f"Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for formeta, score in zip(metadata, scores):
            entry = {
                "src": formeta["src"],
                "mt": formeta["mt"],
                "score": score,
                "noise_level": formeta["level"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    logger.info("Done!")

if __name__ == "__main__":
    main()
