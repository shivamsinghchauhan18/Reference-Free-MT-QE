
import os
import sys
import argparse
import pandas as pd
import sacrebleu
import logging

# Configure logging to both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_lines(file_path):
    logger.info(f"Loading {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def compute_sacrebleu(hyp_lines, ref_lines):
    logger.info("Computing BLEU and chrF++...")
    bleu = sacrebleu.corpus_bleu(hyp_lines, [ref_lines])
    chrf = sacrebleu.corpus_chrf(hyp_lines, [ref_lines])
    
    # Segment-level scores
    bleu_scores = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyp_lines, ref_lines)]
    chrf_scores = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyp_lines, ref_lines)]
    
    return bleu.score, chrf.score, bleu_scores, chrf_scores

def compute_bertscore(hyp_lines, ref_lines, model_type, num_layers=None):
    logger.info(f"STARTING BERTScore with {model_type}...")
    try:
        from bert_score import score
        logger.info(f"Imported bert_score successfully for {model_type}")
    except ImportError:
        logger.error("Could not import bert_score. Skipping.")
        return [0.0] * len(hyp_lines)
        
    lang = "cs" 
    try:
        # Check if model needs download
        logger.info(f"Calling bert_score.score for {model_type}. This may download the model...")
        P, R, F1 = score(hyp_lines, ref_lines, model_type=model_type, lang=lang, verbose=True, num_layers=num_layers, batch_size=32)
        logger.info(f"FINISHED BERTScore with {model_type}.")
        return F1.numpy().tolist()
    except Exception as e:
        logger.error(f"Failed to compute BERTScore for {model_type}: {e}")
        return [0.0] * len(hyp_lines)

def compute_comet(src_lines, hyp_lines, ref_lines, model_name="Unbabel/wmt22-comet-da"):
    logger.info(f"STARTING COMET with {model_name}...")
    try:
        import torch
        from comet import download_model, load_from_checkpoint
        logger.info("Imported comet successfully")
    except ImportError:
        logger.error("Could not import comet. Skipping.")
        return [0.0] * len(hyp_lines)

    try:
        logger.info(f"Downloading/Loading COMET model {model_name}...")
        model_path = download_model(model_name)
        logger.info(f"Model path: {model_path}")
        model = load_from_checkpoint(model_path)
        
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src_lines, hyp_lines, ref_lines)]
        logger.info("Running COMET prediction...")
        model_output = model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        logger.info("FINISHED COMET.")
        return model_output.scores
    except Exception as e:
        logger.error(f"Failed to compute COMET: {e}")
        return [0.0] * len(hyp_lines)

def main():
    parser = argparse.ArgumentParser(description="Baseline Extraction Script")
    parser.add_argument("--src", default="MT/dev.de-hsb.de", help="Source file provided by user")
    parser.add_argument("--hyp", default="hyp_fewshot_full.txt", help="Hypothesis file")
    parser.add_argument("--ref", default="MT/dev.de-hsb.hsb", help="Reference file")
    parser.add_argument("--output", default="results_baseline.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    logger.info("Loading Data...")
    if not os.path.exists(args.src) or not os.path.exists(args.hyp) or not os.path.exists(args.ref):
        logger.error("One or more input files not found. Check paths.")
        return

    src_lines = load_lines(args.src)
    hyp_lines = load_lines(args.hyp)
    ref_lines = load_lines(args.ref)
    
    min_len = min(len(src_lines), len(hyp_lines), len(ref_lines))
    if len(src_lines) != len(hyp_lines) or len(src_lines) != len(ref_lines):
        logger.warning(f"Line count mismatch! Truncating to {min_len} lines.")
        src_lines = src_lines[:min_len]
        hyp_lines = hyp_lines[:min_len]
        ref_lines = ref_lines[:min_len]
        
    results = pd.DataFrame({
        "segment_id": range(min_len),
        "src": src_lines,
        "hyp": hyp_lines,
        "ref": ref_lines
    })
    
    # 1. Traditional Metrics
    sys_bleu, sys_chrf, seg_bleu, seg_chrf = compute_sacrebleu(hyp_lines, ref_lines)
    results["bleu"] = seg_bleu
    results["chrf"] = seg_chrf
    
    # Save intermediate results
    logger.info(f"Saving intermediate traditional metrics to {args.output}...")
    results.to_csv(args.output, index=False)
    
    print("\n" + "="*30)
    print("TRADITIONAL METRICS COMPLETE")
    print("="*30)
    print(f"System BLEU: {sys_bleu:.2f}")
    print(f"System chrF++: {sys_chrf:.2f}")
    print("-" * 20)
    
    # 2. neural - BERTScore
    results["bertscore_xlmr"] = compute_bertscore(hyp_lines, ref_lines, "xlm-roberta-large")
    results.to_csv(args.output, index=False) # Update
    
    results["bertscore_roberta"] = compute_bertscore(hyp_lines, ref_lines, "roberta-large")
    results.to_csv(args.output, index=False) # Update
    
    results["bertscore_glot500"] = compute_bertscore(hyp_lines, ref_lines, "cis-lmu/glot500-base", num_layers=12)
    results.to_csv(args.output, index=False) # Update
    
    # 3. Neural - COMET
    results["comet"] = compute_comet(src_lines, hyp_lines, ref_lines)
    
    # Export Final
    logger.info(f"Saving FINAL results to {args.output}...")
    results.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "="*30)
    print("BASELINE EXTRACTION SUMMARY")
    print("="*30)
    print(f"System BLEU: {sys_bleu:.2f}")
    print(f"System chrF++: {sys_chrf:.2f}")
    print("-" * 20)
    print(f"Avg BERTScore (XLM-R):   {results['bertscore_xlmr'].mean():.4f}")
    print(f"Avg BERTScore (RoBERTa): {results['bertscore_roberta'].mean():.4f}")
    print(f"Avg BERTScore (Glot500): {results['bertscore_glot500'].mean():.4f}")
    print("-" * 20)
    print(f"Avg COMET: {results['comet'].mean():.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
