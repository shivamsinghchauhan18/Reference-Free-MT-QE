
import os
import json
import glob
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files
METRICX_PATH = "evaluation_results_metricx.jsonl"
BASELINE_PATH = "results_baseline.csv"
GPT5_PATH = "LLM-as-a-Judge/800MB_unquantized/4000_wmt25_dev_output_800MB_unquantized_fairseq_sotra_de2hsb_scores_only_gpt-5-mini_reasoning-prompt.txt"

def load_metricx(path):
    data = []
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            data.append({
                "segment_id": i,
                "metricx_score": float(entry.get("predicted_score", 0.0))
            })
    return pd.DataFrame(data)

def load_baseline(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'segment_id' not in df.columns:
        df['segment_id'] = range(0, len(df))
    return df

def load_gpt5(path):
    print(f"Loading GPT-5 from: {path}")
    if not os.path.exists(path):
        print(f"GPT-5 file not found: {path}")
        return pd.DataFrame()
    # Assuming tab-separated based on inspection
    try:
        df = pd.read_csv(path, sep='\t')
        print(f"Loaded GPT-5 raw: {len(df)} rows. Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading GPT-5 CSV: {e}")
        return pd.DataFrame()

    # Check columns
    if 'llm_score' in df.columns:
        # Segment IDs in file might be 1-based, need to align if necessary
        if 'segment_id' in df.columns:
            df['segment_id'] = df['segment_id'] - 1
        else:
             df['segment_id'] = range(0, len(df))
             
        df = df.rename(columns={'llm_score': 'gpt5_score'})
        print(f"GPT-5 processed: {len(df)} rows.")
        return df[['segment_id', 'gpt5_score']]
    print("Column 'llm_score' not found in GPT-5 file.")
    return pd.DataFrame()

def minmax_scale(series):
    return (series - series.min()) / (series.max() - series.min())

def plot_ridge_distributions(df):
    """
    Create a Ridge Plot (Joyplot style) for overlapping distributions.
    """
    # 1. Select Metrics
    metrics = {
        "metricx_score": "MetricX-25",
        "comet": "COMET-DA",
        "bleu": "BLEU",
        "bertscore_xlmr": "BERT-XLMR",
        "chrf": "chrF++",
        "gpt5_score": "GPT-5 Judge"
    }
    
    # 2. Reshape Data for Seaborn (Long Format)
    plot_data = []
    
    for col, label in metrics.items():
        if col in df.columns:
            # Normalize to 0-1 for comparison
            values = minmax_scale(df[col])
            for val in values:
                plot_data.append({
                    "Score (Normalized [0-1])": val,
                    "Metric": label
                })
                
    df_plot = pd.DataFrame(plot_data)
    
    if df_plot.empty:
        print("No data to plot.")
        return

    # 3. Plotting - Overlapping KDE (Ridgeline-eque)
    plt.figure(figsize=(12, 8))
    
    # Option A: Standard Overlapping KDE
    ax = sns.kdeplot(
        data=df_plot, 
        x="Score (Normalized [0-1])", 
        hue="Metric", 
        fill=True, 
        common_norm=False, 
        palette="crest", 
        alpha=0.5, 
        linewidth=1.5
    )
    
    plt.title("Normalized Score Distributions across 4000 Segments", fontsize=16)
    plt.xlabel("Normalized Score (0=Min, 1=Max)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # Add annotations for means?
    # No, keep it clean as requested "single frame"
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "metric_distributions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

    # Option B: Separate Ridge Plot (if they want them vertically stacked)
    # Let's generate both to give the user a choice or include best in report.
    # The user asked for "single frame", overlapping KDE is usually best for comparing shapes directly.
    # But let's look at the vertical stacking too as it handles many metrics well without clutter.
    
    g = sns.FacetGrid(df_plot, row="Metric", hue="Metric", aspect=10, height=1.2, palette="crest")
    g.map(sns.kdeplot, "Score (Normalized [0-1])", clip_on=False, shade=True, alpha=0.8, lw=1.5, bw_adjust=0.5)
    g.map(sns.kdeplot, "Score (Normalized [0-1])", clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Score (Normalized [0-1])")
    
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    
    save_path_ridge = os.path.join(OUTPUT_DIR, "metric_distributions_ridge.png")
    plt.savefig(save_path_ridge, dpi=300)
    plt.close()
    print(f"Saved {save_path_ridge}")

def main():
    print("Loading data...")
    df_mx = load_metricx(METRICX_PATH)
    df_base = load_baseline(BASELINE_PATH)
    df_gpt5 = load_gpt5(GPT5_PATH)
    
    # Merge
    if not df_mx.empty and not df_base.empty:
        df_merged = pd.merge(df_base, df_mx, on="segment_id", how="inner")
        
        # Merge GPT-5 if available
        if not df_gpt5.empty:
            print(f"Merging GPT-5 Data ({len(df_gpt5)} rows)...")
            df_merged = pd.merge(df_merged, df_gpt5, on="segment_id", how="left")
            
        print(f"Merged Data: {len(df_merged)} rows.")
        
        print("Generating Distribution Plots...")
        plot_ridge_distributions(df_merged)
    else:
        print("Error: Could not load data.")

if __name__ == "__main__":
    main()
