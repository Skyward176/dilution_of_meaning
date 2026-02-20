from transformers import PromptDepthAnythingConfig
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import sys
import time
from transformers import TextStreamer
from transformers import pipeline

model = "Qwen/Qwen3-0.6B"
# model = "google/gemma-3n-E2B-it"
#model = "ServiceNow-AI/Apriel-1.5-15b-Thinker"

device = "cuda:0"
if torch.backends.mps.is_available():
    device = torch.device("mps")

file = open("./texts/1984_1.txt")
pipe= pipeline("text-generation", model=model,device=device)
input = file.read()
streamer = TextStreamer(pipe.tokenizer, skip_prompt=True)
for i in range(1,100):
    summarize = [
        {'role':'system', 'content':'Rewrite this passage in your own words and in a similar length and style.'},
        {'role':'user', 'content':input}
    ]
    output = pipe(
        summarize,
        max_new_tokens=2000,
        return_full_text=False,
        do_sample=True,
        temperature=0.7,
        tokenizer_encode_kwargs={"enable_thinking": False},
    )
    print(f'Iteration number {i}\n {output}')
    input = output[0]["generated_text"]