import json
import openai
import faiss
import numpy as np
from tqdm import tqdm
import tiktoken
import os

openai.api_key = ""

# Load JSONL file
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append({"id": obj["id"], "sent": obj["sent"]})
    return data

# Load all JSONL files from a directory (including subdirectories)
def load_all_jsonl_from_folder(folder_path):
    all_data = []
    for root, _, files in os.walk(folder_path):
        print("files", files)
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                all_data.extend(load_jsonl(file_path))
    return all_data

model = "text-embedding-3-small"
# Count tokens using tiktoken
def count_tokens(texts, model=model):
    enc = tiktoken.encoding_for_model(model)
    return sum(len(enc.encode(text)) for text in texts)

# Embed sentences in batches
# model cost : text-embedding-3-small < text-embedding-ada-002 < text-embedding-3-small
# 0.00002$/k token
# full test dataset: 0,235$
def get_embeddings(texts, model=model):
    embeddings = []
    for i in tqdm(range(0, len(texts), 100)):
        batch = texts[i:i+100]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings).astype("float32")

# Load train and test data
train_data = load_all_jsonl_from_folder("train_data_folder")
test_data = load_jsonl("test_dataset.jsonl")

train_sentences = [item["sent"] for item in train_data]
train_ids = [item["id"] for item in train_data]

test_sentences = [item["sent"] for item in test_data]
test_ids = [item["id"] for item in test_data]

# Print token counts
train_tokens = count_tokens(train_sentences)
test_tokens = count_tokens(test_sentences)
print(f"Total tokens in train: {train_tokens}, test: {test_tokens}")
print(f"Total preis in train: {train_tokens* 0.00002} cent, test: {test_tokens* 0.00002 } cent")

# Embed training data
train_embeddings = get_embeddings(train_sentences)
faiss.normalize_L2(train_embeddings)

# Build FAISS index
dimension = train_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(train_embeddings)

# Embed test data
test_embeddings = get_embeddings(test_sentences)
faiss.normalize_L2(test_embeddings)

# Search for top 10 similar train sentence ids for each test sentence
D, I = index.search(test_embeddings, 10)

# Create results as a dictionary
results_dict = {}
for test_idx, neighbors in enumerate(I):
    similar_ids = [train_ids[i] for i in neighbors]
    results_dict[test_ids[test_idx]] = similar_ids

# Save results to JSON file
with open("similarity_results.json", "w", encoding='utf-8') as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)


# Create expanded results with similarity scores
results_with_scores = {}
for test_idx, (neighbors, scores) in enumerate(zip(I, D)):
    similar_items = [
        {"id": train_ids[i], "score": float(similarity)}
        for i, similarity in zip(neighbors, scores)
    ]
    results_with_scores[test_ids[test_idx]] = similar_items

# Save expanded results to JSON file
with open("similarity_results_with_score.json", "w", encoding='utf-8') as f:
    json.dump(results_with_scores, f, indent=2, ensure_ascii=False)