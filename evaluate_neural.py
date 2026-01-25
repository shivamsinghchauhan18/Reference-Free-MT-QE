import argparse
import torch
from bert_score import score
from comet import download_model, load_from_checkpoint
import os
import csv

def load_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def evaluate_bertscore(cands, refs, model_type, lang="cs", num_layers=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Calculating BERTScore on {device} using {model_type}...")

    P, R, F1 = score(
        cands,
        refs,
        model_type=model_type,
        lang=lang,
        device=device,
        verbose=True,
        num_layers=num_layers
    )
    return F1.mean().item()


def evaluate_comet(src, cands, refs, model_path=None):
    print("Calculating COMET score...")

    if model_path:
        model = load_from_checkpoint(model_path)
    else:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, cands, refs)]

    model_output = model.predict(
        data,
        batch_size=8,
        gpus=0,          # MUST be 0
        num_workers=0    # MUST be 0 on macOS
    )

    return model_output.system_score


def evaluate_comet_qe(src, cands, model_name="Unbabel/wmt22-cometkiwi-da"):
    print(f"Calculating COMET QE score ({model_name})...")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    
    data = [{"src": s, "mt": m} for s, m in zip(src, cands)]
    model_output = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0, num_workers=2)
    return model_output.scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate translations using Neural Metrics (BERTScore, COMET).")
    parser.add_argument("--src", required=False, help="Path to source file (required for COMET)")
    parser.add_argument("--hyp", required=False, help="Path to hypothesis file")
    parser.add_argument("--ref", required=False, help="Path to reference file")
    parser.add_argument("--model_path", default=None, help="Path to custom COMET model checkpoint (e.g. LoRA adapter)")
    parser.add_argument("--update_csv", help="Path to CSV file to run QE on and update results.")
    
    args = parser.parse_args()

    if args.update_csv:
        print(f"Updating CSV file: {args.update_csv}")
        rows = []
        with open(args.update_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
        
        if 'wmt22-cometkiwi-da' in fieldnames:
             print("Column 'wmt22-cometkiwi-da' already exists. Skipping or overwriting...")
             # Optionally, you might want to exit or continue. Let's continue and overwrite.
        else:
             fieldnames.append('wmt22-cometkiwi-da')
        
        src_texts = [row['src'] for row in rows]
        hyp_texts = [row['hyp'] for row in rows]
        
        qe_scores = evaluate_comet_qe(src_texts, hyp_texts)
        
        for row, score_val in zip(rows, qe_scores):
            row['wmt22-cometkiwi-da'] = score_val
            
        with open(args.update_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"Successfully updated {args.update_csv} with wmt22-cometkiwi-da scores.")
        return

    if not args.src or not args.hyp or not args.ref:
        print("Error: --src, --hyp, and --ref are required unless --update_csv is used.")
        return

    print(f"Loading data...")
    src_lines = load_lines(args.src)
    hyp_lines = load_lines(args.hyp)
    ref_lines = load_lines(args.ref)

    # Sanity check
    if not (len(src_lines) == len(hyp_lines) == len(ref_lines)):
        print(f"Error: File line counts do not match!")
        print(f"Src: {len(src_lines)}, Hyp: {len(hyp_lines)}, Ref: {len(ref_lines)}")
        return

    print("-" * 50)
    
    # 1. BERTScore - mBERT
    f1_mbert = evaluate_bertscore(hyp_lines, ref_lines, "bert-base-multilingual-cased", lang="cs")
    print(f"BERTScore (mBERT, lang=cs): {f1_mbert:.4f}")
    print("-" * 50)

    # 2. BERTScore - XLM-R
    f1_xlmr = evaluate_bertscore(hyp_lines, ref_lines, "xlm-roberta-base", lang="cs")
    print(f"BERTScore (XLM-R, lang=cs): {f1_xlmr:.4f}")
    print("-" * 50)

    # 3. BERTScore - Glot500
    f1_glot = evaluate_bertscore(hyp_lines, ref_lines, "cis-lmu/glot500-base", lang="cs", num_layers=12)
    print(f"BERTScore (Glot500, lang=cs): {f1_glot:.4f}")
    print("-" * 50)

    # 3. COMET
    comet_score = evaluate_comet(src_lines, hyp_lines, ref_lines, model_path=args.model_path)
    if args.model_path:
        print(f"COMET (Custom): {comet_score:.4f}")
    else:
        print(f"COMET (wmt22-comet-da): {comet_score:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
