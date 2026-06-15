# scripts/prepare_data.py
# Converts glaive-function-calling-v2 to JSON Extraction format
# Run: python scripts/prepare_data.py

import json, re, random, os
from datasets import load_dataset
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────
DATASET_NAME   = 'glaiveai/glaive-function-calling-v2'
OUTPUT_DIR     = 'data'
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
TEST_RATIO     = 0.10
MIN_EXAMPLES   = 2000
TARGET_TOTAL   = 5000   # cap to keep training fast on free GPU
RANDOM_SEED    = 42

# ── INSTRUCTION TEMPLATES ──────────────────────────────────
# Multiple templates so the model generalises, not memorises
INSTRUCTIONS = [
    "Extract the key information from the following text and return it as a valid JSON object.",
    "Parse the input text and output a structured JSON with all relevant fields.",
    "Read the text below and extract entities, values, and parameters as JSON.",
    "Identify and extract all important data from the text. Return valid JSON only.",
    "Convert the following natural language text into a structured JSON object.",
]

def extract_functioncall_json(chat_text):
    """Extract the JSON from <functioncall> blocks in chat text."""
    pattern = r'<functioncall>\s*({.*?})(?:\s*<\|endoftext\|>|$)'
    matches = re.findall(pattern, chat_text, re.DOTALL)
    results = []
    for m in matches:
        try:
            parsed = json.loads(m)
            results.append(parsed)
        except json.JSONDecodeError:
            pass
    return results

def extract_user_message(chat_text):
    """Extract the first USER message from chat text."""
    pattern = r'USER:\s*(.*?)(?:\n\n|ASSISTANT:|$)'
    match = re.search(pattern, chat_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def clean_output_json(parsed_json):
    """Flatten and clean the extracted JSON for training."""
    if not isinstance(parsed_json, dict):
        return None
    # Skip if only contains 'name' with no arguments
    if set(parsed_json.keys()) == {'name'}:
        return None
    # If has 'arguments' key, merge it up one level
    if 'arguments' in parsed_json and isinstance(parsed_json['arguments'], dict):
        result = {}
        if 'name' in parsed_json:
            result['function'] = parsed_json['name']
        result.update(parsed_json['arguments'])
        return result
    return parsed_json

def convert_example(row):
    """Convert one raw glaive row to JSON extraction format."""
    chat = row.get('chat', '')
    user_msg = extract_user_message(chat)
    fc_jsons = extract_functioncall_json(chat)
    if not user_msg or not fc_jsons:
        return None
    cleaned = clean_output_json(fc_jsons[0])
    if not cleaned:
        return None
    # Pick a random instruction template
    instruction = random.choice(INSTRUCTIONS)
    return {
        'instruction': instruction,
        'input': user_msg,
        'output': json.dumps(cleaned, ensure_ascii=False),
    }

def validate_example(ex):
    """Check example is clean and valid."""
    if not ex:
        return False
    if not ex['input'] or len(ex['input']) < 10:
        return False
    try:
        json.loads(ex['output'])
    except:
        return False
    if ex['output'] in ('{}', '[]', 'null'):
        return False
    return True

def format_for_training(ex):
    """Format into the instruction-input-output text format."""
    return {
        'instruction': ex['instruction'],
        'input': ex['input'],
        'output': ex['output'],
        # Full text field for SFTTrainer
        'text': (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )
    }

def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split='train')
    print(f'Loaded {len(ds)} raw examples')

    print("Converting to JSON extraction format...")
    converted = []
    for row in tqdm(ds):
        ex = convert_example(row)
        if validate_example(ex):
            converted.append(format_for_training(ex))
        if len(converted) >= TARGET_TOTAL:
            break

    print(f'Valid examples after conversion: {len(converted)}')
    if len(converted) < MIN_EXAMPLES:
        print(f'WARNING: Only {len(converted)} examples. Check dataset or lower MIN_EXAMPLES.')

    random.seed(RANDOM_SEED)
    random.shuffle(converted)

    n = len(converted)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train   = converted[:n_train]
    val     = converted[n_train:n_train + n_val]
    test    = converted[n_train + n_val:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(OUTPUT_DIR, f'{split_name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Saved {len(split_data)} examples to {path}')

    print("\nData preparation complete!")
    print(f'  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')

if __name__ == '__main__':
    main()