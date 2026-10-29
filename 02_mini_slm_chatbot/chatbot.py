import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype = torch.float32).to(device)
model.eval()

def generate_response(user_message: str) -> str:
    messages=[
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant"
        },
        {
            "role" : "user",
            "content": user_message
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt = True,
        return_tensors='pt',
        return_dict = True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 150,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )
        
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    
    return tokenizer.decode(new_tokens, skip_special_tokens=True)