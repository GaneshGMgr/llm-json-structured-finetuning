# scripts/evaluate.py
# Usage: python scripts/evaluate.py --split test --model_path models/sft_lora

import json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def build_prompt(ex):
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n"
    )

def generate_output(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for eval consistency
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def compute_metrics(predictions, ground_truths):
    valid_json, exact_match, total_fields, present_fields = 0, 0, 0, 0
    n = len(predictions)
    for pred, truth in zip(predictions, ground_truths):
        try:
            pred_json = json.loads(pred)
            valid_json += 1
            truth_json = json.loads(truth)
            # Exact match
            if pred_json == truth_json:
                exact_match += 1
            # Field recall
            if isinstance(truth_json, dict):
                truth_keys = set(truth_json.keys())
                pred_keys  = set(pred_json.keys()) if isinstance(pred_json, dict) else set()
                total_fields   += len(truth_keys)
                present_fields += len(truth_keys & pred_keys)
        except:
            pass
    return {
        'json_validity_rate':  round(valid_json / n, 4),
        'exact_match_accuracy': round(exact_match / n, 4),
        'field_recall':         round(present_fields / total_fields, 4) if total_fields else 0,
        'total_samples':        n,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',      default='test')
    parser.add_argument('--model_path', default=None,  help='Path to LoRA adapter. None=base model')
    parser.add_argument('--base_model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    # Load data
    data = load_jsonl(f'data/{args.split}.jsonl')[:args.max_samples]
    print(f'Evaluating on {len(data)} examples...')

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_path:
        model = PeftModel.from_pretrained(model, args.model_path)
        print(f'Loaded adapter from {args.model_path}')
    else:
        print('Evaluating BASE model (no adapter)')

    # Generate
    predictions, ground_truths = [], []
    for ex in tqdm(data):
        prompt = build_prompt(ex)
        pred   = generate_output(model, tokenizer, prompt)
        predictions.append(pred)
        ground_truths.append(ex['output'])

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    label = 'base' if not args.model_path else 'sft'
    out_path = f'results/{label}_metrics.json'
    import os; os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics saved to {out_path}')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()
