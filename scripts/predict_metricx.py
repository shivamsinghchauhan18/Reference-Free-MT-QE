import argparse
import os
import json
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import lightning.pytorch as pl
from tqdm import tqdm

# Local modules (run from the repo root: python -m scripts.predict_metricx)
from models.metricx_model import MetricX25Evaluator
from models.metricx_dataset import SorbianDataset
from training.train_metricx import MetricXLightningModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run MetricX-25 Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to the pending Lightning checkpoint (.ckpt)")
    parser.add_argument("--src", required=True, help="Path to source text file")
    parser.add_argument("--mt", required=True, help="Path to hypothesis (MT) text file")
    parser.add_argument("--ref", help="Path to reference text file (optional, for correlation)")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", default="google/gemma-3-12b-it", help="Base model name used in training")
    
    args = parser.parse_args()
    
    # 1. Prepare Data
    logger.info("Reading input files...")
    with open(args.src, 'r', encoding='utf-8') as f:
        src_lines = [l.strip() for l in f]
    with open(args.mt, 'r', encoding='utf-8') as f:
        mt_lines = [l.strip() for l in f]
        
    if len(src_lines) != len(mt_lines):
        logger.warning(f"Length mismatch: SRC ({len(src_lines)}) vs MT ({len(mt_lines)}). Truncating to minimum.")
        min_len = min(len(src_lines), len(mt_lines))
        src_lines = src_lines[:min_len]
        mt_lines = mt_lines[:min_len]
    
    # Optional Reference for Correlation
    ref_lines = None
    if args.ref:
        with open(args.ref, 'r', encoding='utf-8') as f:
            ref_lines = [l.strip() for l in f]
        if len(ref_lines) > len(src_lines):
             ref_lines = ref_lines[:len(src_lines)]
        elif len(ref_lines) < len(src_lines):
            logger.error("Reference file is shorter than source/mt. Cannot calculate correlation accurately.")
            ref_lines = None # Disable correlation if mismatch
            
    # Create Temporary JSONL for Dataset
    temp_data_path = "temp_predict_data.jsonl"
    logger.info(f"Creating temporary dataset at {temp_data_path}...")
    with open(temp_data_path, 'w', encoding='utf-8') as f:
        for i, (s, m) in enumerate(zip(src_lines, mt_lines)):
            # Dummy score, we are predicting
            entry = {"src": s, "mt": m, "score": 0.0}
            f.write(json.dumps(entry) + "\n")
            
    # 2. Load Model
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    # Load from Lightning Checkpoint
    # We need to ensure the hparams are compatible. 
    # If the checkpoint has hparams, they override init args.
    model = MetricXLightningModule.load_from_checkpoint(
        args.checkpoint,
        model_name=args.model_name, # Fallback if not in hparams, but usually ignored if in hparams
        map_location="cpu", # Initial load to CPU (Lightning moves to GPU usually, but with QLoRA/BitsAndBytes we might need care)
        strict=False # Allow minor mismatches if necessary
    )
    
    # Move validation/inference to GPU if available (MetricX25Evaluator handles device_Map="auto" internally for base model)
    # But the LightningModule wrapper needs to be consistent.
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    # 3. Validation Loader
    ds = SorbianDataset(temp_data_path, tokenizer, normalize_scores=False) # No normalization for inference input
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # 4. Inference Loop
    logger.info("Starting inference...")
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                
            # Predict
            # Model forward returns tensor scores
            preds = model(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy().tolist())
            
    # 5. Save Results
    logger.info(f"Saving results to {args.output}")
    results = []
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, pred in enumerate(predictions):
            item = {
                "src": src_lines[i],
                "mt": mt_lines[i],
                "predicted_score": pred
            }
            if ref_lines:
                item["ref"] = ref_lines[i]
                
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            results.append(pred)
            
    # 6. Cleanup
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()
