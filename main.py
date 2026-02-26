import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import umap
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configuration

#GEN_MODEL_ID = "openai-community/gpt2-xl"
GEN_MODEL_ID = "Qwen/Qwen3-0.6B"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
ITERATIONS = 2

BATCH_SIZE = 8  # Adjust based on VRAM capacity

device = "cuda:0"
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

print(f'Using device: {device}')

# Initialize Models
print(f"Loading Generation Model: {GEN_MODEL_ID}...")
pipe = pipeline("text-generation", model=GEN_MODEL_ID, device=device)

print(f"Loading Embedding Model: {EMBED_MODEL_ID}...")
embed_model = SentenceTransformer(EMBED_MODEL_ID, device=device)

# Load Texts
text_files = glob.glob("./texts/*.txt")
num_files = len(text_files)
file_names = [os.path.basename(f) for f in text_files]

print(f"Loaded {num_files} files: {', '.join(file_names)}")

# Initialize Results Structure
results = {name: {'texts': [], 'embeddings': []} for name in file_names}
current_texts = []

for file_path in text_files:
    name = os.path.basename(file_path)
    with open(file_path, "r") as f:
        text = f.read().strip()
    current_texts.append(text)
    results[name]['texts'].append(text)

# Generate Original Embeddings (Batch)
print("Generating original embeddings...")
orig_embeddings = embed_model.encode(current_texts, batch_size=BATCH_SIZE)
for idx, name in enumerate(file_names):
    results[name]['embeddings'].append(orig_embeddings[idx])

# Output file setup
with open("output_txt_all.txt", "w") as f:
    f.write(f"Dilution of Meaning: Batched Experiment (Iterations: {ITERATIONS})\n\n")

# Main Iterative Loop (Batched)
for i in range(1, ITERATIONS + 1):
    print(f"\nIteration {i}/{ITERATIONS}...")
    
    # Prepare batch prompts
    batch_prompts = [
        [
            {'role': 'system', 'content': 'Rewrite this passage in your own words and in a similar length and style.'},
            {'role': 'user', 'content': text}
        ] for text in current_texts
    ]
    
    # Batch Generation
    # Note: transformers pipeline handles batching if batch_size is provided
    outputs = pipe(
        batch_prompts,
        max_new_tokens=2000,
        return_full_text=False,
        do_sample=True,
        temperature=0.7,
        batch_size=BATCH_SIZE,
        tokenizer_encode_kwargs={"enable_thinking": False}
    )
    
    # Extract generated texts
    new_texts = [out[0]["generated_text"] for out in outputs]
    
    # Batch Embedding Generation
    new_embeddings = embed_model.encode(new_texts, batch_size=BATCH_SIZE)
    
    # Update results and current_texts
    with open("output_txt_all.txt", "a") as output_file:
        output_file.write(f"--- Iteration {i} ---\n")
        for idx, name in enumerate(file_names):
            results[name]['texts'].append(new_texts[idx])
            results[name]['embeddings'].append(new_embeddings[idx])
            output_file.write(f"FILE: {name}\n{new_texts[idx]}\n\n")
            
    current_texts = new_texts

print("\nAll processing complete.")

# Visualization logic (Semantic Spread / Population Comparison)
print("Generating visualization...")
all_embeddings_list = []
for name in file_names:
    all_embeddings_list.extend(results[name]['embeddings'])

all_embeddings_arr = np.array(all_embeddings_list)

# Use UMAP for dimensionality reduction
print("Running UMAP projection...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(all_embeddings_arr)

# Extract coordinates for Initial (iter 0) and Final (last iter)
initial_coords = []
final_coords = []
for idx in range(num_files):
    start_idx = idx * (ITERATIONS + 1)
    end_idx = start_idx + (ITERATIONS + 1)
    file_2d = embeddings_2d[start_idx:end_idx]
    initial_coords.append(file_2d[0])
    final_coords.append(file_2d[-1])

initial_coords = np.array(initial_coords)
final_coords = np.array(final_coords)

# Calculate spread (mean distance from centroid)
initial_centroid = np.mean(initial_coords, axis=0)
final_centroid = np.mean(final_coords, axis=0)

initial_spread = np.mean(np.linalg.norm(initial_coords - initial_centroid, axis=1))
final_spread = np.mean(np.linalg.norm(final_coords - final_centroid, axis=1))

print(f"Initial Semantic Spread: {initial_spread:.4f}")
print(f"Final Semantic Spread: {final_spread:.4f}")
print(f"Spread Reduction: {((initial_spread - final_spread) / initial_spread) * 100:.1f}%")

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))

# Plot initial spread
ax.scatter(initial_coords[:, 0], initial_coords[:, 1], color='blue', marker='o', s=100, alpha=0.6, label=f'Original Texts (Spread: {initial_spread:.3f})')
# Plot final spread
ax.scatter(final_coords[:, 0], final_coords[:, 1], color='red', marker='X', s=100, alpha=0.8, label=f'Final Rewrites (Iter {ITERATIONS}, Spread: {final_spread:.3f})')

# Draw lines connecting original to final
for i in range(num_files):
    ax.arrow(initial_coords[i, 0], initial_coords[i, 1], 
             final_coords[i, 0] - initial_coords[i, 0], 
             final_coords[i, 1] - initial_coords[i, 1], 
             color='gray', alpha=0.2, length_includes_head=True, head_width=0.01)
    ax.text(initial_coords[i, 0], initial_coords[i, 1], file_names[i], fontsize=8, alpha=0.7)

ax.set_title(f"Semantic Collapse: Initial vs Final Spread (UMAP Projection)\nModel: {GEN_MODEL_ID} | Iterations: {ITERATIONS}", fontsize=14)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("semantic_collapse_umap_plot.png", dpi=300)
print("Plot saved as semantic_collapse_umap_plot.png")
plt.show()