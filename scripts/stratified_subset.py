import json
import numpy as np
import os
import random

def create_stratified_subset(input_file, output_file, train_ratio=0.65, num_bins=10):
    """
    Creates a stratified subset of the data based on score distribution.
    """
    print(f"Loading data from {input_file}...")
    data = []
    scores = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if "score" in item:
                data.append(item)
                scores.append(float(item["score"]))
    
    if not data:
        print("No data found!")
        return

    # Convert to numpy for binning
    scores_np = np.array(scores)
    
    # Create histogram bins to stratify against
    # This ensures we keep the same "shape" of distribution (low/med/high quality translations)
    counts, bin_edges = np.histogram(scores_np, bins=num_bins)
    
    # Bucket indices for each sample
    bin_indices = np.digitize(scores_np, bin_edges) - 1
    
    selected_data = []
    
    print(f" Stratifying into {num_bins} bins...")
    # Sample from each bin
    for b in range(num_bins):
        # Indices of data points in this bin
        indices_in_bin = np.where(bin_indices == b)[0]
        
        if len(indices_in_bin) == 0:
            continue
            
        # Calculate how many to keep from this bin
        n_keep = int(len(indices_in_bin) * train_ratio)
        
        # Randomly select indices
        selected_indices = np.random.choice(indices_in_bin, size=n_keep, replace=False)
        
        for idx in selected_indices:
            selected_data.append(data[idx])
            
    print(f"Original size: {len(data)}")
    print(f"Subset size: {len(selected_data)} ({len(selected_data)/len(data):.2%})")
    
    # Shuffle the final dataset to avoid bin-ordering
    random.shuffle(selected_data)
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in selected_data:
            f.write(json.dumps(item) + "\n")
            
if __name__ == "__main__":
    # Input is the full original file
    INPUT_PATH = "data/qe_finetune_hsb.jsonl"
    # Output is the new training file
    OUTPUT_PATH = "data/train_metricx_65pct.jsonl"
    
    if os.path.exists(INPUT_PATH):
        create_stratified_subset(INPUT_PATH, OUTPUT_PATH)
    else:
        print(f"Error: {INPUT_PATH} not found.")
