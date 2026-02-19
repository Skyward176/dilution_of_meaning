import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
import time
from transformers import TextStreamer

model = "google/gemma-3n-E4B-it"


tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model).to("cuda:0")

file = "./text.txt"

text = open(file)


if __name__ == "__main__":
    prompt = text.read()    

    messages = [
        {"role": "system", "content": "Your goal is to rewrite the provided text in your own words while preserving its main idea and its nuanced meaning.Do not provide any output outside of your rewritten version. Aim to preserve the legnth of the text."},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, # Essential to start with <|assistant|>
        return_tensors="pt"
    ).to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    start_time = time.time()
    outputs = model.generate(**input_ids, max_new_tokens=2000, streamer=streamer)
    end_time = time.time()
    
    num_tokens = len(outputs[0]) - len(input_ids["input_ids"][0])
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken
    
    print(f"\n\nStats:\nTokens generated: {num_tokens}\nTime taken: {time_taken:.2f}s\nTokens per second: {tokens_per_second:.2f}")
