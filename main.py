import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = "google/gemma-3n-E4B-it"


tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, device_map='auto')

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter prompt: ")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))