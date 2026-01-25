
import os
import json
import glob
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

# File Paths
METRICX_PATH = "evaluation_results_metricx.jsonl"
BASELINE_PATH = "results_baseline.csv"
LLM_ZEROSHOT_DIR = "LLM-as-a-Judge/results_zeroshot_full"
LLM_FEWSHOT_DIR = "LLM-as-a-Judge/results_fewshot_full"
REPORT_PATH = "comparison_report.md"

def load_metricx(path):
    print(f"Loading MetricX results from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            # Aligning with other files which are 0-indexed usually
            data.append({
                "segment_id": i,
                "metricx_score": entry.get("predicted_score"),
                "mt_text": entry.get("mt") # For verification
            })
    return pd.DataFrame(data)

def load_baseline(path):
    print(f"Loading Baseline results from {path}...")
    if not os.path.exists(path):
        print("Warning: Baseline CSV not found.")
        return None
    df = pd.read_csv(path)
    # Ensure segment_id is present or create it
    if 'segment_id' not in df.columns:
        df['segment_id'] = range(1, len(df) + 1)
    return df

def get_llm_files():
    files = []
    # Zero-shot
    if os.path.exists(LLM_ZEROSHOT_DIR):
        files.extend(glob.glob(os.path.join(LLM_ZEROSHOT_DIR, "*.csv")))
    # Few-shot
    if os.path.exists(LLM_FEWSHOT_DIR):
        files.extend(glob.glob(os.path.join(LLM_FEWSHOT_DIR, "*.csv")))
    return files

def compute_correlations(df, pred_col, ref_col):
    if pred_col not in df.columns or ref_col not in df.columns:
        return None, None
    # Drop NaNs
    valid = df[[pred_col, ref_col]].dropna()
    if len(valid) < 2:
        return None, None
    pearson, _ = pearsonr(valid[pred_col], valid[ref_col])
    spearman, _ = spearmanr(valid[pred_col], valid[ref_col])
    return pearson, spearman

def main():
    # 1. Load Data
    df_mx = load_metricx(METRICX_PATH)
    df_base = load_baseline(BASELINE_PATH)
    
    # Merge Baseline if available
    if df_base is not None:
        # Avoid duplicate join columns
        cols_to_use = df_base.columns.difference(df_mx.columns).tolist()
        cols_to_use.append("segment_id")
        df_combined = pd.merge(df_mx, df_base[cols_to_use], on="segment_id", how="left")
    else:
        df_combined = df_mx

    llm_files = get_llm_files()
    results = []

    print(f"Found {len(llm_files)} LLM output files.")

    # Define reference metrics to compare against
    # Ordered by importance/user request
    ref_metrics = ["bleu", "comet", "bertscore_xlmr", "bertscore_roberta", "bertscore_glot500"]
    # Filter to only those present in df_combined
    ref_metrics = [m for m in ref_metrics if m in df_combined.columns]

    # Build Header
    # We want MetricX vs LLM first, then comparisons against refs
    header = "| Model | Mode | MetricX vs LLM (Pearson) | MetricX vs LLM (Spearman) |"
    divider = "|---|---|---|---|"
    
    for ref in ref_metrics:
        header += f" LLM vs {ref} (Pearson) | MetricX vs {ref} (Pearson) |"
        divider += "---|---|"
    
    report_lines = ["# SOTA Comparative Analysis Report\n"]
    report_lines.append("## Correlation Analysis\n")
    report_lines.append(header + "\n")
    report_lines.append(divider + "\n")

    for fpath in llm_files:
        fname = os.path.basename(fpath)
        # Identify Metadata
        mode = "Zero-shot" if "zeroshot" in fpath else "Few-shot"
        model_name = fname.replace("hyp_zeroshot_full_sentence_scores_", "").replace("hyp_fewshot_full_sentence_scores_", "").replace(".csv", "")
        
        try:
            df_llm = pd.read_csv(fpath)
            # Ensure segment_id
            if 'segment_id' not in df_llm.columns:
                 df_llm['segment_id'] = range(0, len(df_llm))
            
            # Merge
            # Prefix LLM columns to avoid collision
            llm_score_col = f"llm_score_{model_name}"
            df_llm_renamed = df_llm.rename(columns={"llm_score": llm_score_col})
            
            # We only really need validation of the merge key and the score
            merge_cols = ["segment_id", llm_score_col]
            
            df_curr = pd.merge(df_combined, df_llm_renamed[merge_cols], on="segment_id", how="inner")
            
            # MetricX vs LLM correlation
            p_mx_llm, s_mx_llm = compute_correlations(df_curr, "metricx_score", llm_score_col)
            
            row = f"| {model_name} | {mode} | {p_mx_llm:.4f} | {s_mx_llm:.4f} |"
            
            # Compare against each reference metric
            for ref in ref_metrics:
                # LLM vs Ref
                p_llm_ref, _ = compute_correlations(df_curr, llm_score_col, ref)
                # MetricX vs Ref (re-computed per row purely to align with that row's subset, though usually full set)
                p_mx_ref, _ = compute_correlations(df_curr, "metricx_score", ref)
                
                # Handle None if missing
                val_llm = f"{p_llm_ref:.4f}" if p_llm_ref is not None else "N/A"
                val_mx = f"{p_mx_ref:.4f}" if p_mx_ref is not None else "N/A"
                
                row += f" {val_llm} | {val_mx} |"
            
            report_lines.append(row + "\n")
            
        except Exception as e:
            # Silence expected errors for aggregate files if needed, or just print
            if "not in index" in str(e):
                 # Likely aggregate file without line-level scores
                 print(f"Skipping aggregate file: {fname}")
            else:
                 print(f"Error processing {fname}: {e}")

    # Write Report
    with open(REPORT_PATH, "w") as f:
        f.writelines(report_lines)
    
    print(f"Analysis complete. Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
