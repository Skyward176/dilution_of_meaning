import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = "google/gemma-3n-E4B-it"


tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, device_map='auto')

if __name__ == "__main__":
    import sys
    import time
    from transformers import TextStreamer

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter prompt: ")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print(f"\nGenerating response for: '{prompt}'\n")
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=200, streamer=streamer)
    end_time = time.time()
    
    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    
    print(f"\n\nStats:\nTokens generated: {num_tokens}\nTime taken: {time_taken:.2f}s\nTokens per second: {tokens_per_second:.2f}")