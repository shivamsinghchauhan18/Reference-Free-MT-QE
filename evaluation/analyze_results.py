import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import argparse

def analyze_results(csv_file):
    print(f"Analyzing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Correlation Analysis
    # Agent score is 0-100, BERTScore is 0-1 (usually). Let's normalize agent score for comparison or just correlate.
    # BERTScore is F1.
    
    if len(df) < 2:
        print("Not enough data for correlation analysis.")
        return

    p_corr, _ = pearsonr(df['bert_score'], df['agent_score'])
    s_corr, _ = spearmanr(df['bert_score'], df['agent_score'])
    
    print("\n=== Correlation Analysis ===")
    print(f"Pearson Correlation: {p_corr:.4f}")
    print(f"Spearman Correlation: {s_corr:.4f}")
    
    # 2. Flagging Hallucinations
    # Condition: High BERTScore (> 0.8) but Low Agent Score (< 50)
    # Note: BERTScore thresholds depend on the model. Glot500 might have different distribution. 
    # Assuming 0.8 is 'high' for now.
    
    suspects = df[(df['bert_score'] > 0.8) & (df['agent_score'] < 50)]
    
    print(f"\n=== Hallucination Suspects ({len(suspects)}) ===")
    if not suspects.empty:
        for i, row in suspects.iterrows():
            print(f"\n[Sample {i}]")
            print(f"Source: {row['src']}")
            print(f"Hypothesis: {row['hyp']}")
            print(f"BERTScore: {row['bert_score']:.4f} | Agent Score: {row['agent_score']}")
            print(f"Rationale: {row['agent_rationale']}")
    else:
        print("No obvious hallucinations flagged based on thresholds.")

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bert_score', y='agent_score', hue='agent_score', palette='viridis')
    plt.title(f'Debate Score vs. BERTScore (n={len(df)})')
    plt.xlabel('BERTScore (Glot500)')
    plt.ylabel('Agent Debate Score (0-100)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Pearson: {p_corr:.2f}\nSpearman: {s_corr:.2f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    output_plot = "debate_vs_bert.png"
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Hybrid Evaluation Results")
    parser.add_argument("csv_file", help="Path to evaluation_report.csv")
    args = parser.parse_args()
    
    analyze_results(args.csv_file)
