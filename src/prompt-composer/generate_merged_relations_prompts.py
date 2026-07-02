import os
from utils import generate_similar_example_prompt, load_json, load_jsonl_to_dict
import json
import random

# Define number of shots
num_shots = 10  # Change this value to 1, 3, 5, 10, etc.

# Define root folder and paths
# dataset_name = "wikidata_tekgen"
# dataset_name = "dbpedia_webnlg"
dataset_name = "webnlg"

project_folder = r"\RE-analysis"
root_folder = r"\RE-analysis\datasets\\" 
data_folder = root_folder + dataset_name
test_folder = os.path.join(data_folder, "test")
ontology_path = os.path.join(data_folder, "ontologies", "merged_ontology.json")
train_folder = os.path.join(data_folder, "train")
similarity_folder = os.path.join(data_folder, "test_train_sent_similarity")

# Create output directories if they don't exist
output_dir = os.path.join(project_folder, "prompts", f"{dataset_name}_prompts", f"{num_shots}shot_prompts")
os.makedirs(output_dir, exist_ok=True)


# Create subfolder for current num_shots
output_subdir = os.path.join(output_dir, f"merged_relations_{num_shots}shot_prompt")
os.makedirs(output_subdir, exist_ok=True)

# Process each test file
for file_name in os.listdir(test_folder):
    if not file_name.endswith(".jsonl"):
        continue
        
    # Get ontology ID from filename (format: ont_<id>_<name>_test.jsonl)
    ontology_id = "_".join(file_name.split("_")[1:-1])  # This will get <id>_<name>
    
    # Load corresponding ontology
    ontology = load_json(ontology_path)
    
    # Load corresponding similarity data
    similarity_file = os.path.join(similarity_folder, f"{ontology_id}_test_train_similarity.json")
    similarity_data = load_json(similarity_file)
    
    # Load corresponding train data
    train_file = os.path.join(train_folder, f"ont_{ontology_id}_train.jsonl")
    train_data = load_jsonl_to_dict(train_file)
    
    # Create output JSONL files in their respective subfolders
    output_file = os.path.join(output_subdir, f"{ontology_id}_merged_relations_{num_shots}shot_prompt.jsonl")
    
    # Process test file
    test_path = os.path.join(test_folder, file_name)
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_sentence = data.get("sent", "")
            test_id = data.get("id", "")  # Get the test sentence ID
            
            # Generate 1-shot prompt
            prompt = generate_similar_example_prompt(
                test_sentence=test_sentence,
                test_sentence_id=test_id,
                num_examples= num_shots,
                ontology=ontology,
                similarity_data=similarity_data,
                train_data=train_data
            )
            
            # Save 3-shot prompt to JSONL file
            with open(output_file, 'a', encoding='utf-8') as out_f:
                json.dump({
                    "id": test_id,
                    "prompt": prompt
                }, out_f)
                out_f.write('\n')

    # Randomly select and print 2 examples from each file
    print(f"\nExample prompts for ontology {ontology_id}:")
    print("\n1-shot examples:")
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines:
            samples = random.sample(lines, min(2, len(lines)))
            for sample in samples:
                data = json.loads(sample)
                print(f"\nID: {data['id']}")
                print(f"Prompt:\n{data['prompt']}")
                print("-" * 80)



