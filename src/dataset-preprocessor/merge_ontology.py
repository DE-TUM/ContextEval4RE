import os
import json
from collections import defaultdict


# Define folder and output file names
# dataset_name = "wikidata_tekgen"
# dataset_name = "dbpedia_webnlg"
dataset_name = "webnlg"
root_folder = f"ContextEval4RE/datasets/{dataset_name}"

input_folder = os.path.join(root_folder, 'ontologies')
output_file = os.path.join(root_folder, 'ontologies', 'merged_ontology.json')
log_file = os.path.join(root_folder, 'ontologies', 'merge_log.txt')

# Storage for merged content
all_concepts = []
all_relations = []

# For tracking duplicates
concept_seen = defaultdict(list)  # qid -> list of filenames
relation_seen = defaultdict(list)  # (pid, domain, range) -> list of filenames

# Storage for log information
log_lines = []

# Process each JSON file
for filename in os.listdir(input_folder):
    if filename.endswith("_ontology.json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        concepts = data.get("concepts", [])
        relations = data.get("relations", [])

        for concept in concepts:
            key = concept.get("qid")
            if key:
                concept_seen[key].append(filename)
            all_concepts.append(concept)

        for relation in relations:
            key = (relation.get("pid"), relation.get("domain"), relation.get("range"))
            if key[0]:  # pid should exist
                relation_seen[key].append(filename)
            all_relations.append(relation)

        log_lines.append(f"{filename}: {len(concepts)} concepts, {len(relations)} relations")

# Log number before removing duplicates
log_lines.append(f"Before duplicate removal: {len(all_concepts)} concepts, {len(all_relations)} relations")

# Detect and log duplicates
concept_duplicates = {qid: files for qid, files in concept_seen.items() if len(files) > 1}
relation_duplicates = {key: files for key, files in relation_seen.items() if len(files) > 1}

if concept_duplicates:
    log_lines.append("Duplicate concepts detected:")
    for qid, files in concept_duplicates.items():
        log_lines.append(f"  Concept {qid} found in {', '.join(files)}")

if relation_duplicates:
    log_lines.append("Duplicate relations detected:")
    for key, files in relation_duplicates.items():
        pid, domain, range_ = key
        log_lines.append(f"  Relation (pid={pid}, domain={domain}, range={range_}) found in {', '.join(files)}")

# Remove duplicates
unique_concepts = {concept["qid"]: concept for concept in all_concepts if "qid" in concept}
unique_relations = {(relation["pid"], relation["domain"], relation["range"]): relation for relation in all_relations if "pid" in relation}

# Final merged data
merged_data = {
    "concepts": list(unique_concepts.values()),
    "relations": list(unique_relations.values())
}

# Log number after removing duplicates
log_lines.append(f"After duplicate removal: {len(merged_data['concepts'])} concepts, {len(merged_data['relations'])} relations")

# Write the merged ontology
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

# Write the log
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f"Done! Merged file saved as '{output_file}' and log saved as '{log_file}'.")
