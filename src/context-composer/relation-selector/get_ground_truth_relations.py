import os
import json
from preprocess_utils import load_json, load_jsonl_to_dict

# get n ground truth ontology relations
def get_gt_ontology_relations(ground_truth_data:dict, test_sentence_id:str) -> list:
    gt_relations = []
    triples = None
    # Get the ground truth data for the test ID
    if test_sentence_id in ground_truth_data:
        triples = ground_truth_data[test_sentence_id].get('triples')       

    for triple in triples:
        gt_relations.append(triple['rel'])

    # remove duplicates
    gt_relations = list(set(gt_relations))
    return gt_relations

def process_test_files(test_folder: str, output_folder: str, ground_truth_folder: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each test file
    for file_name in os.listdir(test_folder):
        if not file_name.endswith(".jsonl"):
            continue
            
        # Get ontology ID from filename (format: ont_<id>_<name>_test.jsonl)
        ontology_id = "_".join(file_name.split("_")[1:-1])  # This will get <id>_<name>
        

        # Load ground truth
        ground_truth_path = os.path.join(ground_truth_folder, f"ont_{ontology_id}_ground_truth.jsonl")
        ground_truth_data = load_jsonl_to_dict(ground_truth_path)
        
        # Create output file path
        output_file = os.path.join(output_folder, f"{ontology_id}_gt_relations.jsonl")
        
        # Load test data
        test_path = os.path.join(test_folder, file_name)
        test_data = load_jsonl_to_dict(test_path)
        
        # Process each test example
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for test_id in test_data:
                # Get ground truth relations
                gt_relations = get_gt_ontology_relations(ground_truth_data, test_id)
                
                # Save to output file
                json.dump({
                    "id": test_id,
                    "gt_relations": gt_relations
                }, out_f)
                out_f.write('\n')
                
        print(f"Processed {file_name} -> {os.path.basename(output_file)}")

if __name__ == "__main__":
    # Define paths
    # dataset_name = "dbpedia_webnlg"
    # dataset_name = "wikidata_tekgen"
    dataset_name = "webnlg"
    root_folder = "/datasets" + dataset_name
    test_folder = os.path.join(root_folder, "test")
    ground_truth_folder = os.path.join(root_folder, "ground_truth")
    output_folder = os.path.join(root_folder, "test_ground_truth_relations")
    
    # Process files
    process_test_files(test_folder, output_folder, ground_truth_folder)
