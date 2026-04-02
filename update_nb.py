import json

with open("dilution_of_meaning.ipynb", "r") as f:
    nb = json.load(f)

new_source = """text_files = glob.glob("./texts/*.txt")
iterations = 100
results = {os.path.basename(f): {'texts': [], 'embeddings': []} for f in text_files}
current_texts = []
file_names = []

# Clear or create output.txt
with open("output_txt_all.txt", "w") as f:
    f.write("Dilution of Meaning: All Texts Experiment\\n\\n")

for file_path in text_files:
    file_name = os.path.basename(file_path)
    file_names.append(file_name)
    with open(file_path, "r") as f:
        text = f.read().strip()
    current_texts.append(text)
    
    with open("output_txt_all.txt", "a") as output_file:
        output_file.write(f"--- FILE: {file_name} ---\\n")
        output_file.write(f"Original:\\n{text}\\n\\n")

# Generate original embeddings (SentenceTransformers natively supports batching)
print("Generating original embeddings...")
orig_embeddings = embed_model.encode(current_texts, batch_size=len(current_texts))
for i, file_name in enumerate(file_names):
    results[file_name]['texts'].append(current_texts[i])
    results[file_name]['embeddings'].append(orig_embeddings[i])

# Batch size for LLM generation - adjust this based on your available memory
# Using len(current_texts) to run as many as possible in parallel
batch_size = len(current_texts) 

for i in range(1, iterations + 1):
    print(f"\\nRunning iteration {i}")
    
    summarize_prompts = [
        [
            {'role': 'system', 'content': 'Summarize the provided text capturing all necessary details, content, and tone in 1 to 3 sentences.'},
            {'role': 'user', 'content': text}
        ]
        for text in current_texts
    ]
    
    # Generate summaries in parallel
    print("Summarizing...")
    summaries_output = sum(
        summarize_prompts,
        max_new_tokens=2000,
        return_full_text=False,
        do_sample=True,
        temperature=0.7,
        batch_size=batch_size,
        tokenizer_encode_kwargs={"enable_thinking": False},
    )
    
    summaries = [out[0]["generated_text"] for out in summaries_output]
    
    generate_prompts = [
        [
            {'role': 'system', 'content': 'Pretend you are the author of the text described in the provided summary. Generate a text of around 500 words in line with the provided summary.'},
            {'role': 'user', 'content': summary}
        ]
        for summary in summaries
    ]
    
    # Generate new texts in parallel
    print("Generating new texts...")
    regenerated_output = gen(
        generate_prompts,
        max_new_tokens=2000,
        return_full_text=False,
        do_sample=True,
        temperature=0.7,
        batch_size=batch_size,
        tokenizer_encode_kwargs={"enable_thinking": False},
    )
    
    generated_texts = [out[0]["generated_text"] for out in regenerated_output]
    
    # Generate embeddings in parallel
    print("Generating embeddings...")
    new_embeddings = embed_model.encode(generated_texts, batch_size=len(generated_texts))
    
    # Update state and write to file
    with open("output_txt_all.txt", "a") as output_file:
        for idx, file_name in enumerate(file_names):
            summary = summaries[idx]
            gen_text = generated_texts[idx]
            
            results[file_name]['texts'].append(gen_text)
            results[file_name]['embeddings'].append(new_embeddings[idx])
            
            output_file.write(f"--- FILE: {file_name} ---\\n")
            output_file.write(f"Summary:\\n{summary}\\n\\n")
            output_file.write(f"Iteration {i}:\\n{gen_text}\\n\\n")
            
    current_texts = generated_texts

print("\\nAll processing complete.")"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if isinstance(cell["source"], list):
            content = "".join(cell["source"])
        else:
            content = cell["source"]
            
        if "text_files = glob.glob(\\"./texts/*.txt\\")" in content and "for file_path in tqdm(text_files, desc=\\"Processing Files\\"):" in content:
            new_lines = [line + "\\n" for line in new_source.split("\\n")]
            if new_lines[-1].endswith("\\n"):
               new_lines[-1] = new_lines[-1][:-1]
            cell["source"] = new_lines
            break

with open("dilution_of_meaning.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
