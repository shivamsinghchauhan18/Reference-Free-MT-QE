
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    csv_file = "results_baseline.csv"
    model_name = "Unbabel/wmt22-comet-da"
    
    logger.info(f"Loading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logger.error(f"File {csv_file} not found!")
        return

    logger.info(f"Loading COMET model: {model_name}...")
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
    except Exception as e:
        logger.error(f"Failed to load COMET: {e}")
        return

    logger.info("Preparing data for COMET...")
    data = [
        {"src": row["src"], "mt": row["hyp"], "ref": row["ref"]}
        for _, row in df.iterrows()
    ]
    
    logger.info("Computing COMET scores...")
    # gpus=0 forces CPU if CUDA not available/compatible on Mac MPS for some ops
    # Using num_workers=2 to avoid "multiprocessing_context" error with num_workers=0
    model_output = model.predict(data, batch_size=8, gpus=0, num_workers=2) 
    
    df["comet"] = model_output.scores
    
    logger.info(f"Saving updated results to {csv_file}...")
    df.to_csv(csv_file, index=False)
    
    print("\n" + "="*30)
    print("COMET METRICS COMPLETE")
    print("="*30)
    print(f"Avg COMET: {df['comet'].mean():.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
