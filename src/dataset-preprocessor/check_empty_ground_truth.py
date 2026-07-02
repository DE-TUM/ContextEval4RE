import os
import json

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set your folder path
folder_path = os.path.join(current_dir, 'ground_truth')
log_file_path = os.path.join(current_dir, 'empty_triples_log.txt')

# Initialize the log content
log_lines = []

# Loop through all .jsonl files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(folder_path, filename)
        empty_ids = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if not item.get("triples"):  # Check if triples is empty
                    empty_ids.append(item["id"])
        
        # Log file summary
        log_lines.append(f"File: {filename}")
        log_lines.append(f"Number of items with empty triples: {len(empty_ids)}")
        log_lines.append("IDs:")
        log_lines.extend(empty_ids)
        log_lines.append("")  # Add a blank line between files

# Write the log to a file
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    log_file.write('\n'.join(log_lines))

print(f"Logging complete. Output saved to {log_file_path}")
