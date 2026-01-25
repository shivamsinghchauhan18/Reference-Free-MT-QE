import os
import random
import argparse
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "openai/gpt-4o-mini"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def get_few_shot_examples(train_src, train_tgt, n=3):
    """Selects n random examples from the training data."""
    indices = random.sample(range(len(train_src)), min(n, len(train_src)))
    examples = []
    for i in indices:
        examples.append(f"German: {train_src[i]}\nHigh Sorbian: {train_tgt[i]}")
    return "\n\n".join(examples)

def construct_prompt(line, train_src, train_tgt, strategy, n_shots):
    if strategy == "zeroshot":
        return f"""You are an expert translator from German to High Sorbian.
Translate the following German sentence to High Sorbian.
Output ONLY the translation, no other text.

German: {line}
High Sorbian:"""

    elif strategy == "fewshot":
        examples = get_few_shot_examples(train_src, train_tgt, n_shots)
        return f"""You are an expert translator from German to High Sorbian.
Translate the following German sentence to High Sorbian.
Use the provided examples as a guide for style and vocabulary.
Output ONLY the translation, no other text.

Examples:
{examples}

German: {line}
High Sorbian:"""

    elif strategy == "cot":
        return f"""You are an expert translator from German to High Sorbian.
Translate the following German sentence to High Sorbian.
First, think step-by-step about the grammar and vocabulary of the sentence.
Then, provide the final translation starting with "Translation: ".

German: {line}
"""
    return ""

def parse_response(response_text, strategy):
    if strategy == "cot":
        # Look for "Translation: ..."
        match = re.search(r"Translation:\s*(.*)", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # Fallback: try to take the last line if it looks like a translation
            lines = response_text.strip().split('\n')
            return lines[-1].strip()
    else:
        return response_text.strip()

def translate_line(line, train_src, train_tgt, strategy="fewshot", n_shots=3):
    prompt = construct_prompt(line, train_src, train_tgt, strategy, n_shots)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates German to High Sorbian."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )
        content = response.choices[0].message.content
        return parse_response(content, strategy)
    except Exception as e:
        print(f"Error translating line: {line}\nError: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Translate German to High Sorbian using LLM.")
    parser.add_argument("input_file", help="Path to the input German file")
    parser.add_argument("output_file", help="Path to save the translated file")
    parser.add_argument("--train_src", default="MT/train.de-hsb.de", help="Path to training source file")
    parser.add_argument("--train_tgt", default="MT/train.de-hsb.hsb", help="Path to training target file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of lines to translate")
    parser.add_argument("--strategy", choices=["zeroshot", "fewshot", "cot"], default="fewshot", help="Prompting strategy")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples")
    
    args = parser.parse_args()

    train_src = []
    train_tgt = []
    if args.strategy == "fewshot":
        print("Loading training data...")
        train_src = load_data(args.train_src)
        train_tgt = load_data(args.train_tgt)
        print(f"Loaded {len(train_src)} training pairs.")

    print(f"Reading input file: {args.input_file}")
    input_lines = load_data(args.input_file)
    
    if args.limit:
        print(f"Limiting translation to first {args.limit} lines.")
        input_lines = input_lines[:args.limit]

    translated_lines = []
    print(f"Starting translation using strategy: {args.strategy}...")
    
    for i, line in enumerate(input_lines):
        if (i + 1) % 10 == 0:
            print(f"Translating line {i+1}/{len(input_lines)}...")
        translation = translate_line(line, train_src, train_tgt, args.strategy, args.shots)
        translated_lines.append(translation)

    print(f"Saving translations to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
