# scripts/generate_dpo_pairs.py
# Creates 'chosen' vs 'rejected' preference pairs for DPO
# Run after SFT training is complete

import json, random
from tqdm import tqdm

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def degrade_json(good_json_str):
    """Create a bad version of a good JSON output."""
    try:
        obj = json.loads(good_json_str)
    except:
        return good_json_str
    if not isinstance(obj, dict):
        return good_json_str
    keys = list(obj.keys())
    if len(keys) < 2:
        return '{}'
    degradations = [
        # Remove half the keys
        lambda o, k: {k2: v for k2, v in o.items() if k2 != k[0]},
        # Rename a key to something wrong
        lambda o, k: dict(list({f'unknown_{k[0]}': o[k[0]]}.items()) + list({k2: v for k2, v in o.items() if k2 != k[0]}.items())),
        # Make all values empty strings
        lambda o, k: {k2: '' for k2 in list(o.keys())[:len(k)//2]},
    ]
    fn = random.choice(degradations)
    try:
        return json.dumps(fn(obj, keys), ensure_ascii=False)
    except:
        return '{}'

def build_prompt(ex):
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

def main():
    data = load_jsonl('data/train.jsonl')
    pairs = []
    for ex in tqdm(data):
        prompt   = build_prompt(ex)
        chosen   = ex['output']       # the good, clean JSON
        rejected = degrade_json(ex['output'])  # degraded version
        if chosen == rejected:
            continue
        pairs.append({
            'prompt':   prompt,
            'chosen':   chosen,
            'rejected': rejected,
        })
    out_path = 'data/dpo_pairs.jsonl'
    with open(out_path, 'w') as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Created {len(pairs)} DPO pairs -> {out_path}')

if __name__ == '__main__':
    main()
