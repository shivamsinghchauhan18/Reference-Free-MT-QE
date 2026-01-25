
import os
import json
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Files
METRICX_PATH = "evaluation_results_metricx.jsonl"
BASELINE_PATH = "results_baseline.csv"
LLM_ZEROSHOT_DIR = "LLM-as-a-Judge/results_zeroshot_full"
LLM_FEWSHOT_DIR = "LLM-as-a-Judge/results_fewshot_full"
REPORT_PATH = "meta_evaluation_report.md"

def load_metricx(path):
    print(f"Loading MetricX from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            data.append({
                "segment_id": i,
                "metricx_score": float(entry.get("predicted_score", 0.0))
            })
    return pd.DataFrame(data)

def load_baseline(path):
    print(f"Loading Baseline from {path}...")
    df = pd.read_csv(path)
    if 'segment_id' not in df.columns:
        df['segment_id'] = range(0, len(df))
    return df

def load_llm_scores():
    print("Loading LLM Judge Scores...")
    files = []
    if os.path.exists(LLM_ZEROSHOT_DIR):
        files.extend(glob.glob(os.path.join(LLM_ZEROSHOT_DIR, "*.csv")))
    if os.path.exists(LLM_FEWSHOT_DIR):
        files.extend(glob.glob(os.path.join(LLM_FEWSHOT_DIR, "*.csv")))
    
    llm_data = {} # model_name -> DataFrame
    
    for fpath in files:
        try:
            fname = os.path.basename(fpath)
            # Skip aggregates (heuristic)
            if "aggregate" in fname:
                continue
                
            model_name = fname.replace("hyp_zeroshot_full_sentence_scores_", "").replace("hyp_fewshot_full_sentence_scores_", "").replace(".csv", "")
            mode = "Zero-shot" if "zeroshot" in fpath else "Few-shot"
            display_name = f"{model_name} ({mode})"
            
            df = pd.read_csv(fpath)
            if 'segment_id' not in df.columns:
                df['segment_id'] = range(0, len(df))
            
            # Normalize column name
            if 'llm_score' in df.columns:
                 llm_data[display_name] = df[['segment_id', 'llm_score']].rename(columns={'llm_score': display_name})
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            
    return llm_data

def z_score_normalize(series):
    return (series - series.mean()) / series.std()

def main():
    # 1. Load Data
    df_mx = load_metricx(METRICX_PATH)
    df_base = load_baseline(BASELINE_PATH)
    llm_dfs = load_llm_scores()
    
    # 2. Merge All Data
    # Base is the anchor (contains segment_id, hyp, ref, comet)
    merged_df = df_base.copy()
    
    # Merge MetricX
    merged_df = pd.merge(merged_df, df_mx, on="segment_id", how="inner")
    
    # Merge LLMs
    for name, df in llm_dfs.items():
        merged_df = pd.merge(merged_df, df, on="segment_id", how="left")
        
    print(f"Total Rows after merging: {len(merged_df)}")
    
    # 3. Define Proxy & Candidates
    # PROXY: COMET (High-quality reference-based metric, best proxy for human DA)
    proxy_col = "comet"
    
    candidates = ["metricx_score", "bleu", "chrf", "bertscore_xlmr", "bertscore_roberta", "bertscore_glot500"]
    # Add LLM columns
    candidates.extend(llm_dfs.keys())
    
    # 4. Normalize & Correlate
    results = []
    
    # Apply Z-score to everything for "Cross-Metric Comparability" (User Request)
    # Note: Correlations are invariant to linear scaling, but we'll do it to satisfy the requirement 
    # and maybe print statistics.
    
    norm_df = merged_df.copy()
    cols_to_norm = [proxy_col] + [c for c in candidates if c in merged_df.columns]
    
    for c in cols_to_norm:
        norm_df[c] = z_score_normalize(norm_df[c])
        
    print("Computed Z-Scores for all metrics.")
    
    # Compute Correlations
    for cand in candidates:
        if cand not in norm_df.columns:
            continue
            
        # Filter NaNs
        valid = norm_df[[cand, proxy_col]].dropna()
        if len(valid) < 2:
            continue
            
        p_r, _ = pearsonr(valid[cand], valid[proxy_col])
        s_rho, _ = spearmanr(valid[cand], valid[proxy_col])
        
        results.append({
            "Metric": cand,
            "Pearson (r)": p_r,
            "Spearman (rho)": s_rho
        })
        
    # 5. Determine Champion
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Pearson (r)", ascending=False)
    
    champion = results_df.iloc[0]
    
    # 6. Generate Report
    with open(REPORT_PATH, "w") as f:
        f.write("# Advanced Meta-Evaluation Report (Ticket #3)\n\n")
        f.write("## Objective\n")
        f.write("Identify the 'Champion Metric' by analyzing correlations with the **COMET** metric (Human-Proxy).\n")
        f.write("All scores were **Z-Score Normalized** prior to analysis.\n\n")
        
        f.write("## Champion Metric\n")
        f.write(f"The best performing metric is **{champion['Metric']}** with a Pearson correlation of **{champion['Pearson (r)']:.4f}**.\n\n")
        
        f.write("## Correlation Matrix (vs COMET Proxy)\n")
        f.write(results_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        f.write("## Key Insights\n")
        f.write("- **MetricX-25**: How does it compare to reference metrics?\n")
        f.write("- **LLM Judges**: Do they align with the neural proxy?\n")
        
    print(f"Meta-evaluation complete. Report written to {REPORT_PATH}")
    print(results_df)

if __name__ == "__main__":
    main()
