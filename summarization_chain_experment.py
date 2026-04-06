import torch
import matplotlib
# from mlx_vlm import load, generate
from mlx_lm import load, generate
from mlx_lm import sample_utils
from pathlib import Path
from bert_score import score



def _generate(model, tokenizer, prompt, temperature: float, max_tokens: int):
    sampler = sample_utils.make_sampler(temp=temperature)
    logits_processors = sample_utils.make_logits_processors()
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        sampler=sampler,
        logits_processors=logits_processors,
        max_tokens=max_tokens
    )


def summarize_subroutine(text, tokenizer, model):
    tokenized = tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content':
                    'Summarize the provided text in 50 to 60 words.'},
                {'role': 'user', 'content': text},
            ],
            add_generation_prompt=True,
            enable_thinking=False
        )
    summarized_output = _generate(
                                 model,
                                 tokenizer,
                                 prompt=tokenized,
                                 temperature=0.7,
                                 max_tokens=1000,
                                 )
    return summarized_output


if __name__ == "__main__":
    # Batching isn't working in mlx-vlm for gemma4 rn
    # BATCH_SIZE = 32

    INPUT_DIR = "./paragraphs"
    ITERATIONS = 5
    dir = Path(INPUT_DIR)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # gen_model_id = "mlx-community/gemma-4-e4b-it-8bit"

    gen_model_id = "mlx-community/Qwen3-0.6B-bf16"
    gen, gen_tokenizer = load(gen_model_id)

    print("Loading data . . .")
    texts = []
    for text in dir.iterdir():
        if not text.is_dir() and text.name != ".DS_Store":
            if text.is_file():
                with open(text, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read().strip())

    output = []
    print("Commencing summarization loop: ")
    for text in texts:
        for i in range(ITERATIONS):
            text = summarize_subroutine(text, gen_tokenizer, gen)
        output.append(text)

    print(output)
    candidates = texts
    references = output

    # lang="en" automatically picks the best model (usually RoBERTa-large)
    P, R, F1 = score(candidates, references, lang="en", verbose=True)

    print(f"F1 Score: {F1.mean():.4f}")
