import torch
import matplotlib.pyplot as plt
# from mlx_vlm import load, generate
from mlx_lm import load, generate
from mlx_lm import sample_utils
from pathlib import Path
from bert_score import BERTScorer
import numpy as np
from sentence_transformers import SentenceTransformer, util
import kagglehub


class SemanticScorer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Initializes the model once and pins it to the specified device.
        """
        # Automatically detect CUDA (GPU), otherwise fallback to CPU
        if device is None:
            self.device = torch.device("cpu")
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            if torch.cuda.is_available():
                self.device = torch.device("cuda")

            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def score(self, source, target):
        """
        Calculates the cosine similarity between a source and target string.
        """
        # Encode strings into dense vector embeddings
        # convert_to_tensor ensures the math happens on the GPU
        embeddings = self.model.encode([source, target],
                                       convert_to_tensor=True)
        # Compute cosine similarity
        # We use .item() to convert the single-element tensor to a float
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()

    def score_batch(self, sources, targets):
        """
        Efficiently scores multiple pairs at once.
        'sources' and 'targets' should be lists of equal length.
        """
        if len(sources) != len(targets):
            raise ValueError(
                    "Source and Target lists must be the same length."
                    )
        # Bulk encoding is significantly faster than individual calls
        emb1 = self.model.encode(sources, convert_to_tensor=True)
        emb2 = self.model.encode(targets, convert_to_tensor=True)
        # This computes a similarity matrix; we take the diagonal for
        # pair-wise scores
        # cosine_scores[i][j] is the similarity between emb1[i] and emb2[j]
        cosine_scores = util.cos_sim(emb1, emb2)
        return torch.diagonal(cosine_scores).tolist()


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

    path = kagglehub.dataset_download("nishantsingh96/refined-bookcorpus-dataset")

    print("Path to dataset files:", path)
    INPUT_DIR = "./paragraphs"
    ITERATIONS = 100
    dir = Path(INPUT_DIR)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("Loading bert scorer")
    bertscorer = BERTScorer(lang="en", device=device)

    print("Loading semanticscorer")
    semanticscorer = SemanticScorer(device=device)

    # gen_model_id = "mlx-community/gemma-4-e4b-it-8bit"

    gen_model_id = "mlx-community/Qwen3-0.6B-bf16"
    gen, gen_tokenizer = load(gen_model_id)

    print("Loading data . . .")
    texts = []
    text_names = []
    for text in dir.iterdir():
        if not text.is_dir() and text.name != ".DS_Store":
            if text.is_file():
                with open(text, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read().strip())
                    text_names.append(text.name)  # Store the file name

    output = []
    print("Commencing summarization loop: ")
    bertscores = []
    semanticscores = []
    for text in texts:
        text_semanticscores = []
        text_bertscores = []
        original = text
        for i in range(ITERATIONS):
            text = summarize_subroutine(text, gen_tokenizer, gen)
            precision, recall, F1 = bertscorer.score(refs=[original],
                                                     cands=[text])
            text_semanticscores.append(semanticscorer.score(original, text))
            text_bertscores.append(F1)
        output.append(text)
        bertscores.append(text_bertscores)
        semanticscores.append(text_semanticscores)

    # Plot F1 score for each text over iterations
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot BERTScore F1 scores
    ax1.set_title('Iteration vs BERTScore F1 Score')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('BERTScore F1')
    for text, scores in enumerate(bertscores):
        ax1.plot(range(ITERATIONS), scores, label=text_names[text])
    ax1.legend()
    # Plot Semantic Similarity scores
    ax2.set_title('Iteration vs Semantic Similarity Score')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Semantic Similarity')
    for text, scores in enumerate(semanticscores):
        ax2.plot(range(ITERATIONS), scores, label=text_names[text])
    ax2.legend()
    plt.tight_layout()
    plt.show()
