import os
import argparse
import re
from openai import OpenAI
from dotenv import load_dotenv
from statistics import mean, median, stdev
from typing import List, Optional, Tuple

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "openai/gpt-4o-mini"

def load_data(file_path):
    """Load text file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def construct_judge_prompt(source_lang: str, target_lang: str, source_seg: str, target_seg: str) -> str:
    """Construct the LLM judge prompt based on the reference template."""
    return f"""Score the following translation from {source_lang} to
{target_lang} on a scale from 0 to 100, where a score of 0 means a
broken or poor translation; 33 indicates a flawed translation
with significant issues; 66 indicates a good translation with
only minor issues in grammar, fluency, or consistency; and 100
represents a perfect translation in both meaning and grammar.

Answer with only a whole number representing the score, and
nothing else.

{source_lang} source text:

{source_seg}

{target_lang} translation:

{target_seg}"""

def extract_score(response_text: str) -> Optional[int]:
    """Extract the numeric score from the LLM response."""
    # Try to find a number between 0-100
    match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', response_text.strip())
    if match:
        score = int(match.group(1))
        # Clamp to valid range
        return max(0, min(100, score))
    else:
        # Fallback: try to find any number
        numbers = re.findall(r'\d+', response_text.strip())
        if numbers:
            score = int(numbers[0])
            return max(0, min(100, score))
        print(f"Warning: Could not extract score from response: {response_text[:100]}")
        return None

def judge_segment(source_lang: str, target_lang: str, source_seg: str, target_seg: str) -> Optional[int]:
    """Judge a single translation segment using LLM."""
    prompt = construct_judge_prompt(source_lang, target_lang, source_seg, target_seg)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of translation quality."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50  # Score should be short
        )
        content = response.choices[0].message.content
        return extract_score(content)
    except Exception as e:
        print(f"Error judging segment: {e}")
        return None

def evaluate_hypothesis(source_file: str, reference_file: str, hypothesis_file: str, 
                       source_lang: str = "German", target_lang: str = "High Sorbian") -> Tuple[List[int], dict]:
    """Evaluate a hypothesis file against reference."""
    print(f"\nEvaluating: {hypothesis_file}")
    print("=" * 60)
    
    source_lines = load_data(source_file)
    reference_lines = load_data(reference_file)
    hypothesis_lines = load_data(hypothesis_file)
    
    # Validate line counts
    if len(source_lines) != len(reference_lines):
        print(f"Warning: Source ({len(source_lines)}) and reference ({len(reference_lines)}) have different line counts.")
    
    if len(hypothesis_lines) != len(reference_lines):
        print(f"Warning: Hypothesis ({len(hypothesis_lines)}) and reference ({len(reference_lines)}) have different line counts.")
    
    min_lines = min(len(source_lines), len(reference_lines), len(hypothesis_lines))
    scores = []
    
    for i in range(min_lines):
        if (i + 1) % 10 == 0:
            print(f"Judging segment {i+1}/{min_lines}...")
        
        score = judge_segment(source_lang, target_lang, source_lines[i], hypothesis_lines[i])
        if score is not None:
            scores.append(score)
        else:
            print(f"Warning: Failed to get score for segment {i+1}")
    
    # Calculate statistics
    if scores:
        stats = {
            'mean': mean(scores),
            'median': median(scores),
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
        if len(scores) > 1:
            stats['stdev'] = stdev(scores)
        else:
            stats['stdev'] = 0.0
    else:
        stats = {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'stdev': 0.0, 'count': 0}
    
    return scores, stats

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translation quality using LLM as a judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single hypothesis file
  python judge_llm.py MT/dev.de-hsb.de MT/dev.de-hsb.hsb output.txt
  
  # Evaluate all 3 strategies
  python judge_llm.py MT/dev.de-hsb.de MT/dev.de-hsb.hsb \\
      --hypothesis output_zeroshot.txt output_fewshot.txt output_cot.txt \\
      --names zeroshot fewshot cot
        """
    )
    
    parser.add_argument("source_file", help="Path to source language file (German)")
    parser.add_argument("reference_file", help="Path to reference translation file (High Sorbian)")
    parser.add_argument("hypothesis_file", nargs='?', help="Path to hypothesis translation file (optional if using --hypothesis)")
    parser.add_argument("--hypothesis", nargs='+', help="Path(s) to hypothesis translation file(s) - can specify multiple")
    parser.add_argument("--names", nargs='+', help="Names/labels for hypothesis files (for output display)")
    parser.add_argument("--source-lang", default="German", help="Source language name (default: German)")
    parser.add_argument("--target-lang", default="High Sorbian", help="Target language name (default: High Sorbian)")
    parser.add_argument("--output", help="Path to save detailed scores (optional)")
    
    args = parser.parse_args()
    
    # Determine which hypothesis files to evaluate
    hypothesis_files = []
    if args.hypothesis:
        hypothesis_files = args.hypothesis
    elif args.hypothesis_file:
        hypothesis_files = [args.hypothesis_file]
    else:
        parser.error("Must provide either hypothesis_file or --hypothesis")
    
    # Get names for display
    if args.names:
        if len(args.names) != len(hypothesis_files):
            parser.error("Number of --names must match number of hypothesis files")
        names = args.names
    else:
        # Use filenames as names
        names = [os.path.basename(f) for f in hypothesis_files]
    
    # Validate source and reference files exist
    if not os.path.exists(args.source_file):
        parser.error(f"Source file '{args.source_file}' not found.")
    
    if not os.path.exists(args.reference_file):
        parser.error(f"Reference file '{args.reference_file}' not found.")
    
    # Evaluate each hypothesis file
    all_results = {}
    
    for hyp_file, name in zip(hypothesis_files, names):
        if not os.path.exists(hyp_file):
            print(f"Error: Hypothesis file '{hyp_file}' not found. Skipping.")
            continue
        
        scores, stats = evaluate_hypothesis(
            args.source_file, 
            args.reference_file, 
            hyp_file,
            args.source_lang,
            args.target_lang
        )
        
        all_results[name] = {
            'scores': scores,
            'stats': stats,
            'file': hyp_file
        }
        
        # Print statistics
        print(f"\nResults for {name}:")
        print(f"  Mean score:   {stats['mean']:.2f}")
        print(f"  Median score: {stats['median']:.2f}")
        print(f"  Min score:    {stats['min']}")
        print(f"  Max score:    {stats['max']}")
        print(f"  Std dev:      {stats['stdev']:.2f}")
        print(f"  Segments:     {stats['count']}")
    
    # Print comparison if multiple files
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<20} {'Mean':<10} {'Median':<10} {'Std Dev':<10}")
        print("-" * 60)
        for name, result in all_results.items():
            stats = result['stats']
            print(f"{name:<20} {stats['mean']:<10.2f} {stats['median']:<10.2f} {stats['stdev']:<10.2f}")
    
    # Save detailed scores if requested
    if args.output:
        print(f"\nSaving detailed scores to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            for name, result in all_results.items():
                f.write(f"# {name}\n")
                f.write(f"# File: {result['file']}\n")
                f.write(f"# Mean: {result['stats']['mean']:.2f}, Median: {result['stats']['median']:.2f}\n")
                for i, score in enumerate(result['scores'], 1):
                    f.write(f"{i}\t{score}\n")
                f.write("\n")
        print("Done.")

if __name__ == "__main__":
    main()
