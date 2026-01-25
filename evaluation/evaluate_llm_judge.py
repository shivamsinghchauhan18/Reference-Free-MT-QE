import os
import argparse
import re
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "x-ai/grok-4.1-fast:free"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def construct_evaluation_prompt(src, hyp, ref):
    return f"""You are an expert evaluator of machine translation quality.

Source (German): {src}
Reference (High Sorbian): {ref}
Hypothesis (High Sorbian): {hyp}

Evaluate the quality of the hypothesis translation compared to the reference and the source.
Rate it from 1 to 100 with two order of decimal places (100 being perfect, 1 being completely wrong).
Consider accuracy, fluency, and grammar. It's very important to rate the translation based on these criteria.
Sematics Similarity: 
Context Similarity: 
Grammatical Correctness:    
Orthographic Correctness: 
Fluency: 

Format your response exactly as follows:
Score: [Score]
Reasoning: [Reasoning]
"""

def parse_evaluation(response_text):
    # Try to find "Score: <number>"
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", response_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def evaluate_line(src, hyp, ref):
    prompt = construct_evaluation_prompt(src, hyp, ref)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates translation quality."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Low temperature for consistent scoring
            max_tokens=256
        )
        content = response.choices[0].message.content
        score = parse_evaluation(content)
        return score, content
    except Exception as e:
        print(f"Error evaluating line: {e}")
        return None, str(e)

def main():
    parser = argparse.ArgumentParser(description="Evaluate translations using LLM-as-a-Judge (Grok).")
    parser.add_argument("--src", required=True, help="Path to source file")
    parser.add_argument("--hyp", required=True, help="Path to hypothesis file")
    parser.add_argument("--ref", required=True, help="Path to reference file")
    parser.add_argument("--output", required=True, help="Path to save detailed evaluation results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of lines to evaluate")

    args = parser.parse_args()

    print("Loading data...")
    src_lines = load_data(args.src)
    hyp_lines = load_data(args.hyp)
    ref_lines = load_data(args.ref)

    if not (len(src_lines) == len(hyp_lines) == len(ref_lines)):
        print("Error: File line counts do not match!")
        if not args.limit:
             return

    if args.limit:
        print(f"Limiting evaluation to first {args.limit} lines.")
        src_lines = src_lines[:args.limit]
        hyp_lines = hyp_lines[:args.limit]
        ref_lines = ref_lines[:args.limit]

    scores = []
    results = []

    print(f"Starting evaluation using model: {MODEL_NAME}...")
    
    for i, (s, h, r) in enumerate(zip(src_lines, hyp_lines, ref_lines)):
        if (i + 1) % 5 == 0:
            print(f"Evaluating line {i+1}/{len(src_lines)}...")
        
        score, reasoning = evaluate_line(s, h, r)
        
        if score is not None:
            scores.append(score)
        
        results.append(f"Line {i+1}:\nSource: {s}\nRef: {r}\nHyp: {h}\nScore: {score}\nReasoning: {reasoning}\n{'-'*50}")

    print(f"Saving detailed results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(res + "\n")

    if scores:
        avg_score = np.mean(scores)
        print(f"\nAverage LLM-as-a-Judge Score: {avg_score:.2f} / 100")
    else:
        print("\nNo valid scores parsed.")

if __name__ == "__main__":
    main()
