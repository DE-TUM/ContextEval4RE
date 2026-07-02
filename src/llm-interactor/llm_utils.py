import json
import re

# load json
def load_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
# load jsonl
def load_jsonl(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

# load jsonl and convert to dict
def load_jsonl_to_dict(file_path):
    """
    Load a JSONL file and convert it into a dictionary where keys are the 'id' values.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        dict: Dictionary with 'id' as keys and the complete data as values
    """
    result_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # Use the 'id' field as the key
            result_dict[data['id']] = data
    return result_dict

def parse_result(result_str):
    """
    Convert result string from format relation(subject, object) to a list of triples [subject, relation, object].
    If the input does not match the expected format, return an empty list.
    """
    triples = []
    pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
    
    matches = re.findall(pattern, result_str)
    
    if not matches:
        return []  # Return empty list if no valid triples are found

    for relation, subject, obj in matches:
        triples.append([subject.strip(), relation.strip(), obj.strip()])
    
    return triples