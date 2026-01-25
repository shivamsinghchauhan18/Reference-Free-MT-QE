import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorbianDataset(Dataset):
    """
    Dataset for MetricX-25 Reference-Free QE (DE-HSB).
    Handles Z-score normalization and MetricX-style input formatting.
    """
    def __init__(
        self, 
        file_path: str, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 2048,
        normalize_scores: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"Loading data from {file_path}...")
        raw_scores = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Ensure we have required fields
                if "mt" not in item:
                    continue
                    
                src = item.get("src", "")
                mt = item.get("mt", "")
                score = float(item.get("score", 0.0))
                
                self.data.append({
                    "src": src,
                    "mt": mt,
                    "score": score
                })
                raw_scores.append(score)
        
        self.raw_scores = np.array(raw_scores)
        
        if normalize_scores and len(self.raw_scores) > 1:
            logger.info("Applying Z-score normalization...")
            self.mean = np.mean(self.raw_scores)
            self.std = np.std(self.raw_scores)
            # Avoid division by zero
            if self.std == 0:
                self.std = 1.0
            
            # Pre-calculate normalized scores
            for item in self.data:
                item["norm_score"] = (item["score"] - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
            for item in self.data:
                item["norm_score"] = item["score"]
                
        logger.info(f"Loaded {len(self.data)} examples. Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["src"]
        mt = item["mt"]
        score = item["norm_score"]
        
        # MetricX-25 V2 Input Format:
        # [SRC] {DE_Source} [TGT] {HSB_Target} [SRC_LANG] de_DE [TGT_LANG] hsb_DE [SCORE_TYPE] DA
        input_text = (
            f"[SRC] {src} "
            f"[TGT] {mt} "
            f"[SRC_LANG] de_DE "
            f"[TGT_LANG] hsb_DE "
            f"[SCORE_TYPE] DA"
        )
        
        # Tokenization
        encoded = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False, # Dynamic padding handling in collate_fn
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float),
            "original_score": torch.tensor(item["score"], dtype=torch.float)
        }

    @staticmethod
    def get_special_tokens() -> List[str]:
        return ["[SRC]", "[TGT]", "[SRC_LANG]", "[TGT_LANG]", "[SCORE_TYPE]"]
