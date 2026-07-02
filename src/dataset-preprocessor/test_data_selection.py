import os
import json
import random


# Configuration
sample_size = 100 # for wikidata_tekgen
# sample_size = 56 # for dbpedia_webnlg

dataset_name = "wikidata_tekgen"

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_folder = current_dir

# Paths
input_folder = os.path.join(root_folder, 'test_filtered')
output_folder = os.path.join(root_folder, 'test')
log_folder = os.path.join(root_folder, 'preprocess_log')
log_file_path = os.path.join(log_folder, 'subset_log.txt')

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok= True)

# Prepare to collect log entries
log_entries = []
total_new_data_count = 0
total_original_data_count = 0

# Loop over each .jsonl file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jsonl'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read all lines from the original file
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_length = len(lines)
        total_original_data_count += original_length

        # If fewer than sample_size lines, keep all lines
        if original_length < sample_size:
            print(f"Warning: {filename} has less than {sample_size} lines. Including all lines.")
            sampled_lines = lines
        else:
            sampled_lines = random.sample(lines, sample_size)

        # Write the sampled or full lines to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)

        new_length = len(sampled_lines)
        total_new_data_count += new_length

        # Add log entry
        log_entries.append(f"{filename}: original = {original_length}, new = {new_length}")

# Add total counts to log
log_entries.append(f"Total data in original dataset: {total_original_data_count}")
log_entries.append(f"Total data in new dataset: {total_new_data_count}")

# Write log to file
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    log_file.write('\n'.join(log_entries))

print("Subset creation complete. Log file generated.")
