import os
import json
import random



def sample_examples_from_jsonl(input_folder: str, output_folder: str, num_samples: int = 10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all .jsonl files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_folder, filename)
            ontology_id = "_".join(filename.split("_")[:-1])  # This will get ont_<number>_<name>
            # Read all lines from the file
            with open(input_path, "r", encoding="utf-8") as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            
            # Check if the file has fewer than 10 data entries
            if len(data) < num_samples:
                print(f"File {filename} has only {len(data)} entries, sampling all available data.")

            # Sample 10 examples (or fewer if the file has less than 10)
            sampled_data = random.sample(data, min(num_samples, len(data)))

            # Prepare output file name
            output_filename = f"{ontology_id}_examples_random.jsonl"
            output_path = os.path.join(output_folder, output_filename)

            # Write sampled data to the new file
            with open(output_path, "w", encoding="utf-8") as out_f:
                for item in sampled_data:
                    out_f.write(json.dumps(item) + "\n")

            print(f"Saved {len(sampled_data)} examples to {output_path}")

if __name__ == "__main__":
    
    dataset_name = "dbpedia_webnlg"
    root_folder = "C:\\Users\\yuche\\Desktop\\RE-analysis\\datasets\\" + dataset_name

    input_dir = os.path.join(root_folder, "train")
    output_dir = os.path.join(root_folder, "examples_random")

    sample_examples_from_jsonl(input_dir, output_dir)
