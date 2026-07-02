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

def get_ontology_id(file_name: str) -> str:
    """
    Extracts the ontology ID from a file name in the format 'ont_<id>_<name>_test.jsonl'.

    Args:
        file_name (str): The input file name.

    Returns:
        str: Extracted ontology ID (e.g., '1_university').
    """
    match = re.match(r"ont_(.*?)_test\.jsonl", file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError("File name does not match expected pattern: 'ont_<id>_<name>_test.jsonl'")

def get_concept_label(ontology_dict, concept):
    """
    Get the label for the ontology concept
    :param ontology: ontology with concepts ids and labels
    :param ont_dom: input concept ID
    :return: the label of the input concept
    """
    for onto in ontology_dict['concepts']:
        if onto['qid'] == concept:
            return onto['label']
        
        
def get_ontology_relations(ontology_dict):
    """
    Generate a verbalized list of relations in the given ontology to be included in the prompt
    :param ontology:  an ontology as a dictionary
    :return: A string for ontology relations.
            e.g. cast_member(film,human), director (film,human), screenwriter(film,human), producer(film,human), ...
    """

    ont_rels = ""
    onto_rel_strings = list()
    for onto in ontology_dict['relations']:
        ont_rel = onto['label']
        ont_rel = ont_rel.replace(" ", "_")
        ont_dom = onto['domain']
        ont_range = onto['range']
        ont_domain = get_concept_label(ontology_dict, ont_dom)
        ont_range = get_concept_label(ontology_dict, ont_range)

        if ont_rel == None:
            continue
        if ont_domain == None:
            ont_domain = ""
        if ont_range == None:
            ont_range = ""

        onto_rel_strings.append(f"{ont_rel} ({ont_domain},{ont_range})\n")

        ont_rels += ont_rel + "(" + ont_domain + "," + ont_range + "), "

    return ont_rels[0:-2]

# get domain and range of a relation    
def get_domain_range(relation: str, ontology: dict):
    """
    Returns the domain and range of a given relation label from the ontology data.
    
    :param relation: The label of the relation (string)
    :param data: The ontology data (dictionary loaded from JSON)
    :return: A tuple (domain, range) or (None, None) if relation not found
    """
    # Create a mapping of qid to label for concepts
    concept_map = {concept["qid"]: concept["label"] for concept in ontology.get("concepts", [])}
    
    for rel in ontology.get("relations", []):
        if rel.get("label") == relation:
            domain_id = rel.get("domain")
            range_id = rel.get("range")
            return concept_map.get(domain_id, None), concept_map.get(range_id, None)
    
    return None, None  # Return None if relation is not found

# get gt relations prompt
def get_gt_relations_prompt(gt_rels: list, ontology: dict):

    gt_relations_prompt = ""
    for rel in gt_rels:
        domain, range = get_domain_range(rel, ontology)
        rel_string = rel.replace(" ", "_")
        if domain == None:
            domain = ""
        if range == None:
            range = ""
        gt_relations_prompt += f"{rel_string}({domain}, {range}), "
    return gt_relations_prompt

def get_example_random(num_examples, example_random_path):
    examples = []

    with open(example_random_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            data = json.loads(line)
            sent = data.get("sent", "")
            triples = data.get("triples", [])
            # Format triples as: rel(sub, obj)
            triples_str = ", ".join(
                [f'{t["rel"]}({t["sub"]}, {t["obj"]})' for t in triples]
            )
            # Build formatted output
            formatted = f"\n\nExample Sentence: {sent} \nExample Triples: {triples_str}"
            examples.append(formatted)

    return "\n\n".join(examples)


def get_example_similar(test_sentence_id, num_examples, similarity_data, train_data):
    examples = []
    
    # Get similar example IDs for the test sentence
    similar_ids = similarity_data.get(test_sentence_id, [])[:num_examples]
    
    # Get examples for each similar ID
    for similar_id in similar_ids:
        # Access the example directly from train_data dictionary using the ID as key
        example = train_data.get(similar_id)
        if example:
            sent = example.get("sent", "")
            triples = example.get("triples", [])
            # Format triples as: rel(sub, obj)
            triples_str = ", ".join(
                [f'{t["rel"]}({t["sub"]}, {t["obj"]})' for t in triples]
            )
            # Build formatted output
            formatted = f"\n\nExample Sentence: {sent} \nExample Triples: {triples_str}"
            examples.append(formatted)
    
    return "\n\n".join(examples)


def get_test_prompt(test_sentence):
    test_prompt = "\n\nTest Sentence: " + test_sentence
    test_prompt += "\nTest Output: "
    return test_prompt

    
def generate_random_example_prompt(
    test_sentence: str,
    num_examples: int,
    ontology: dict,
    example_random_path: str
) -> str:

    instruction = '''Extract relational triplets from the sentence based on the provided ontology relations and examples.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. 
'''

    example_prompt = get_example_random(num_examples, example_random_path)

    prompt = instruction + '\n'
    prompt += 'Ontology Relations: '
    prompt += get_ontology_relations(ontology)
    prompt += example_prompt
    prompt += get_test_prompt(test_sentence)

    return prompt


def generate_similar_example_prompt(
    test_sentence: str,
    test_sentence_id: str,
    num_examples: int,
    ontology: dict,
    similarity_data: dict,
    train_data: dict
) -> str:

    instruction = '''Extract relational triplets from the sentence based on the provided ontology relations and examples.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. 
'''

    example_prompt = get_example_similar(test_sentence_id, num_examples, similarity_data, train_data)

    prompt = instruction + '\n'
    prompt += 'Ontology Relations: '
    prompt += get_ontology_relations(ontology)
    prompt += example_prompt
    prompt += get_test_prompt(test_sentence)

    return prompt

def generate_gt_relations_prompt(
    test_sentence: str,
    test_sentence_id: str,
    num_examples: int,
    gt_rels: list,
    ontology_data: dict,
    similarity_data: dict,
    train_data: dict
):
    instruction = '''Extract relational triplets from the sentence based on the provided ontology relations and examples.
Use only the listed relations and ensure subjects and objects align with their specified restrictions.
Only return the triples in the format relation(subject, object), separated by commas. Do not include explanations, extra text, or comments. 
'''

    gt_relations_prompt = get_gt_relations_prompt(gt_rels, ontology_data)
    example_prompt = get_example_similar(test_sentence_id, num_examples, similarity_data, train_data)

    prompt = instruction + '\n'
    prompt += 'Ontology Relations: '
    prompt += gt_relations_prompt
    prompt += example_prompt
    prompt += get_test_prompt(test_sentence)

    return prompt

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