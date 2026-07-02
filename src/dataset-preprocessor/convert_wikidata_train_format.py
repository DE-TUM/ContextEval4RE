import os
import json

# copy this file into the wikidata folder and run

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
input_folder = os.path.join(current_dir, 'train_original')
output_folder = os.path.join(current_dir, "train")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                data = json.loads(line)
                new_data = {
                    "id": data["id"],
                    "sent": data["sent"],
                    "triples": [
                        {
                            "sub": data["sub_label"],
                            "rel": data["rel_label"],
                            "obj": data["obj_label"]
                        }
                    ]
                }
                outfile.write(json.dumps(new_data) + "\n")
