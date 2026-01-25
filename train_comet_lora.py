import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from comet.models import ReferencelessRegression
from peft import get_peft_model, LoraConfig, TaskType
import json

class QE_Dataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "src": item["src"],
            "mt": item["mt"],
            "score": float(item["score"])
        }

class QE_DataModule(LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=8):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_ds = QE_Dataset(self.train_path)
        self.val_ds = QE_Dataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # Allow the model's own prepare_sample to be called if possible, 
        # but COMET models expect a list of dicts or specific dict format.
        # ReferencelessRegression.forward expects: src_inputs, mt_inputs
        # But usually we pass data to the model's train_step.
        # Let's return a list of dicts which the model (hopefully) handles or we handle in a loop.
        
        # Actually, let's look at how we are running. We are using PyTorch Lightning Trainer.
        # The model's training_step expects 'batch'.
        # COMET's training_step usually expects a dictionary with 'src', 'mt', 'score'.
        
        src = [x["src"] for x in batch]
        mt = [x["mt"] for x in batch]
        score = torch.tensor([x["score"] for x in batch], dtype=torch.float)
        
        return {"src": src, "mt": mt, "score": score}

def train(args):
    # Performance optimization for H100
    torch.set_float32_matmul_precision('medium')

    print(f"Initializing fresh ReferencelessRegression from xlm-roberta-large...")
    # robust fix: avoid gated checkpoints by initializing from public base encoder
    # Pass dummy lists for train_data/validation_data to satisfy base class setup checks
    model = ReferencelessRegression(
        encoder_model="XLM-RoBERTa",
        pretrained_model="xlm-roberta-large",
        learning_rate=args.learning_rate,
        optimizer="AdamW",
        train_data=[args.train_data], 
        validation_data=[args.val_data]
    )

    # 2. Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "key", "value"] 
    )
    
    # Wrap encoder
    model.encoder = get_peft_model(model.encoder, peft_config)
    model.encoder.print_trainable_parameters()

    # 3. Data Module
    print(f"Loading data from {args.train_data}...")
    dm = QE_DataModule(args.train_data, args.val_data, args.batch_size)

    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss", # COMET returns val_loss usually
        mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    # 5. Train
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Unbabel/wmt22-cometkiwi-da", help="Base model checkpoint or name")
    parser.add_argument("--train_data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--val_data", required=True, help="Path to validation data (JSONL)")
    parser.add_argument("--output_dir", default="checkpoints/comet_lora", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    args = parser.parse_args()
    
    # Ensure modules are downloaded/available
    # Since we are using xlm-roberta-large (public), we don't need the complex checkpoint download logic.
        
    train(args)
