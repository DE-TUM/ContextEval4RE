import os
import json

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set your folder path
test_folder = os.path.join(current_dir, 'test_original')
ground_truth_folder = os.path.join(current_dir, 'ground_truth')
output_folder = os.path.join(current_dir, 'test_filtered')
log_file_path = os.path.join(current_dir, 'preprocess_log', 'test_filter_log.txt')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

log_lines = []

# Helper: Load ground truth data into a dict keyed by ID
def load_ground_truth(filepath):
    id_to_triples = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            id_to_triples[item['id']] = item.get('triples', [])
    return id_to_triples

# Process each test file
for test_filename in os.listdir(test_folder):
    if test_filename.endswith('.jsonl'):
        test_path = os.path.join(test_folder, test_filename)

        # Build corresponding ground truth filename
        gt_filename = test_filename.replace('test', 'ground_truth')
        gt_path = os.path.join(ground_truth_folder, gt_filename)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {test_filename}")
            continue

        # Load ground truth triples
        gt_dict = load_ground_truth(gt_path)

        # Process test file
        original_count = 0
        filtered_items = []

        with open(test_path, 'r', encoding='utf-8') as tf:
            for line in tf:
                item = json.loads(line)
                original_count += 1
                gt_triples = gt_dict.get(item['id'], [])
                if gt_triples:  # Only keep if ground truth triples are non-empty
                    filtered_items.append(item)

        # Write filtered data to new test file
        output_path = os.path.join(output_folder, test_filename)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for item in filtered_items:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Log the result
        log_lines.append(f"File: {test_filename}")
        log_lines.append(f"Original count: {original_count}")
        log_lines.append(f"After filtering: {len(filtered_items)}")
        log_lines.append("")

# Write log
with open(log_file_path, 'w', encoding='utf-8') as log_f:
    log_f.write('\n'.join(log_lines))

print(f"Filtering complete. Filtered test files saved to '{output_folder}'. Log saved to '{log_file_path}'.")
