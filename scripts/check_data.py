# scripts/check_data.py  — Run after prepare_data.py
import json, os

def check_split(path):
    valid, invalid = 0, 0
    examples = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                ex = json.loads(line.strip())
                json.loads(ex['output'])  # validate nested JSON
                valid += 1
                examples.append(ex)
            except:
                invalid += 1
    print(f'{path}: {valid} valid, {invalid} invalid')
    if examples:
        print(f'  Sample input:  {examples[0]["input"][:80]}')
        print(f'  Sample output: {examples[0]["output"][:80]}')
    return valid

for split in ['train', 'val', 'test']:
    check_split(f'data/{split}.jsonl')
