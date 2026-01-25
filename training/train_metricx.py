import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from metricx_dataset import SorbianDataset
from metricx_model import MetricX25Evaluator

class MetricXLightningModule(pl.LightningModule):
    def __init__(self, model_name, lr=1e-5, lora_r=16, margin=0.1, lambda_ranking=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize Model
        self.model = MetricX25Evaluator(model_name=model_name)
        
        # Prepare for QLoRA (4-bit training)
        # This casts non-int8 modules to float32 and enables gradient checkpointing
        self.model.backbone = prepare_model_for_kbit_training(self.model.backbone)

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model.backbone = get_peft_model(self.model.backbone, peft_config)
        self.model.backbone.print_trainable_parameters()
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        scores = batch["score"] # Normalized Z-scores
        
        preds = self(input_ids, mask)
        
        # 1. MSE Loss (Regression)
        loss_mse = self.mse_loss(preds, scores)
        
        # 2. Pairwise Margin Ranking Loss (Robustness)
        # We create pairs from the batch by shifting
        # Pair (i, i+1)
        # If batch size is B, we get B-1 pairs
        if preds.size(0) > 1:
            preds_a = preds[:-1]
            preds_b = preds[1:]
            scores_a = scores[:-1]
            scores_b = scores[1:]
            
            # Target is 1 if score_a > score_b, else -1
            target = torch.sign(scores_a - scores_b)
            # Replace 0s with 1s (equal scores treated as A > B or ignored, but sign returns 0)
            target[target == 0] = 1 
            
            loss_rank = self.ranking_loss(preds_a, preds_b, target)
        else:
            loss_rank = 0.0
            
        # Combined Loss
        # Stage 1: Robustness (MSE + Ranking)
        # Stage 2: Calibration (MSE only) -> Controlled by flag or separate run
        # For now we implement the logic based on argument/hyperparam or step count
        # But simpler to just keep ranking always or use a lambda
        
        loss = loss_mse + self.hparams.lambda_ranking * loss_rank
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", loss_mse, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        scores = batch["score"]
        
        preds = self(input_ids, mask)
        loss = self.mse_loss(preds, scores)
        
        # Calculate Pearson/Spearman on batch (approximation) or just MSE
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Performance optimization for H100 (Tensor Cores)
    torch.set_float32_matmul_precision('medium')

    # Verify token exists before starting heavy job
    if not os.getenv("HF_TOKEN"):
        # Fallback: Check if user has logged in via CLI (harder to check from python without hf_hub)
        # But per ticket, we strictly enforce env var for cluster stability
        print("WARNING: HF_TOKEN not found in environment variables.")
        print("Please ensure you have set HF_TOKEN in your .env file or exported it.")
        # We don't raise error immediately to allow CLI auth to work if present, 
        # but we warn strongly.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True) # Usually Gold subset or holdout
    parser.add_argument("--model_name", default="google/gemma-3-12b-it") # Or fallback
    parser.add_argument("--output_dir", default="checkpoints/metricx")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--stage", choices=["robustness", "calibration"], default="robustness")
    
    args = parser.parse_args()
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.model_max_length = 512 # Cap sequence length for memory safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Data
    train_ds = SorbianDataset(args.train_data, tokenizer)
    val_ds = SorbianDataset(args.val_data, tokenizer, normalize_scores=True) # Always normalize val consistently
    
    from transformers import DataCollatorWithPadding
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # Training Stage Logic
    lambda_ranking = 0.1 if args.stage == "robustness" else 0.0
    
    model = MetricXLightningModule(
        model_name=args.model_name,
        lr=args.lr,
        lora_r=args.lora_r,
        lambda_ranking=lambda_ranking
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"metricx-{args.stage}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed",
        callbacks=[checkpoint_callback],
        logger=CSVLogger(args.output_dir, name="logs"),
        accelerator="gpu",
        devices=1
    )
    
    print(f"Starting Stage: {args.stage.upper()}")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
