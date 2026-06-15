import modal
import json

# Define the Modal app
app = modal.App('json-extractor-api')

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'transformers>=4.40', 'peft>=0.10',
        'torch>=2.1', 'accelerate>=0.28', 'fastapi', 'uvicorn'
    )
)

# Model is loaded once and cached on the GPU container
@app.cls(
    gpu=modal.gpu.T4(),          # free tier GPU
    image=image,
    container_idle_timeout=300,  # keep warm for 5 min after last request
)
class JSONExtractorModel:
    MODEL_NAME = 'your-hf-username/qwen-json-extractor'  # ← your merged model
    INSTRUCTION = 'Extract the key information from the text and return it as JSON.'

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print('Loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.float16,
            device_map='auto', trust_remote_code=True
        )
        self.model.eval()
        print('Model ready.')

    @modal.method()
    def extract(self, text: str) -> dict:
        import torch
        prompt = (
            f'### Instruction:\n{self.INSTRUCTION}\n\n'
            f'### Input:\n{text.strip()}\n\n'
            f'### Response:\n'
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=200,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        raw = self.tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()

        # Extract JSON block
        if '{' in raw:
            start = raw.index('{')
            depth, end = 0, start
            for i, ch in enumerate(raw[start:], start):
                if ch=='{': depth+=1
                elif ch=='}': depth-=1
                if depth==0: end=i; break
            raw = raw[start:end+1]

        try:
            return {'success': True, 'data': json.loads(raw), 'raw': raw}
        except:
            return {'success': False, 'data': {}, 'raw': raw,
                    'error': 'Model output is not valid JSON'}


# FastAPI web endpoint
@app.function(image=image)
@modal.web_endpoint(method='POST')
async def extract_json_api(request: dict):
    text = request.get('text', '')
    if not text.strip():
        return {'success': False, 'error': 'No text provided', 'data': {}}
    model = JSONExtractorModel()
    return model.extract.remote(text)
