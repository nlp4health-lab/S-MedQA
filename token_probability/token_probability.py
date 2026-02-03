import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
import random

# Import term extraction function
from term_extraction import extract_terms_from_dataset
import spacy
import scispacy
import en_core_sci_sm
from scispacy.linking import EntityLinker

nlp = en_core_sci_sm.load()

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

def extract_terms_from_prompt(prompt, nlp, linker):

    terms = []
    # Process the sentence with ScispaCy
    doc = nlp(prompt)

    for entity in doc.ents:  # Iterate over extracted entities
        if len(entity._.kb_ents) == 0:
            continue  # Skip entities with no linked concepts in UMLS
        else:
            umls_ent = entity._.kb_ents[0]  # Get the top-ranked linked concept (CUI, probability)
            cui = umls_ent[0]

            if cui in linker.kb.cui_to_entity:
                entity_info = linker.kb.cui_to_entity[cui]
                # tui = linker.kb.cui_to_entity[umls_ent[0]][3][0]
                entity_name = entity_info.canonical_name  # Get the canonical name of the entity
                if entity_name not in terms:
                    # terms.append(entity_name)  # Add unique terms only
                    terms.append(entity.text)
    return terms

def get_token_probabilities(model, tokenizer, prompt, phrases, term_to_keep=None):
    """
    Compute probabilities for the first sub-word of each word in a phrase given a prompt.

    Args:
        model: The model to pass the prompt to.
        tokenizer: The tokenizer associated with the model.
        prompt (str): The context prompt.
        phrases (list of str): List of phrases for which to compute probabilities.
        term_to_keep (set, optional): Set of phrases to keep for processing.

    Returns:
        list of tuples: A list of (phrase, joint_prob) for each phrase.
    """
    prompt_token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=prompt_token_ids)

    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    print(f"logits: {logits}")
    log_probs = torch.log_softmax(logits, dim=-1)
    print(f"log_probs: {log_probs}")

    results = []
    all_phrase_probabilities = {}

    def find_sublist_index(lst, sub_list):
        n, m = len(lst), len(sub_list)
        for i in range(n - m + 1):
            if lst[i:i + m] == sub_list:
                return i
        return -1


    for phrase in phrases:
        if term_to_keep and phrase not in term_to_keep:
            continue

        word_start_indices = []
        phrase_token_ids = tokenizer(" " + phrase, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompt_token_list = prompt_token_ids[0].tolist()
        phrase_token_list = phrase_token_ids[0].tolist()
        start_idx = find_sublist_index(prompt_token_list, phrase_token_list)
        for i in range(len(phrase_token_ids[0])):
            word_start_indices.append(start_idx + i)

        # decoded_term = tokenizer.decode(word_start_indices)
        print(f"sentence: {prompt}")
        print(f"phrase: {phrase}")

        print(f"word_start_indices: {word_start_indices}")
    

        # Calculate joint probability for the token before the first sub-word of each word
        phrase_probs = []
        for idx in word_start_indices:
            token_id = prompt_token_ids[0][idx].item()
            print(f"log_probs to append: {log_probs[0, idx-1, token_id]}")
            phrase_probs.append(log_probs[0, idx-1, token_id].item())
            print(f"phrase probs: {phrase_probs}")

        joint_prob = sum(phrase_probs)
        results.append((phrase, joint_prob))
        all_phrase_probabilities[phrase] = joint_prob

    return results, all_phrase_probabilities

def process_dataset(dataset, adapter, model_name, tokenizer, term_dict, specialty_filter_fn=None):
    """
    Process a single dataset, extracting terms and analyzing token probability changes.

    Args:
        dataset (list): The dataset to process.
        adapter (str): Path to the fine-tuned model adapter.
        model_name (str): Name of the base model.
        tokenizer: The tokenizer associated with the model.
        term_dict (dict): Dictionary mapping terms to specialties.
        specialty_filter_fn (callable, optional): Function to filter specialties.

    Returns:
        dict: Results for each specialty.
    """
    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.eval()

    adapted_model = AutoModelForCausalLM.from_pretrained(model_name)
    adapted_model = PeftModel.from_pretrained(adapted_model, adapter)
    adapted_model.eval()

    # Analyze probabilities
    combined_results = {}
    all_base_probs_to_save = []
    all_adapt_probs_to_save = []

    count = 0
    for entry in dataset:
        prompt = re.sub(r'[\[\]\(\)]', '', entry["instruction"]).lower()
        phrases = extract_terms_from_prompt(prompt, nlp, linker)

        # Filter terms based on the dictionary and the filtering function
        term_to_keep = set(
            phrase for phrase in phrases if phrase in term_dict and \
            (specialty_filter_fn is None or specialty_filter_fn(term_dict[phrase]))
        )
        
        # Compute probabilities
        results_base, probs_base_to_save = get_token_probabilities(base_model, tokenizer, prompt, phrases, term_to_keep)
        results_adapt, probs_adapt_to_save = get_token_probabilities(adapted_model, tokenizer, prompt, phrases, term_to_keep)

        # Store probabilities
        all_base_probs_to_save.append(probs_base_to_save)
        all_adapt_probs_to_save.append(probs_adapt_to_save)


        for (phrase_base, prob_base), (phrase_adapt, prob_adapt) in zip(results_base, results_adapt):
            if phrase_base == phrase_adapt:
                prob_diff = prob_adapt - prob_base
                specialties = term_dict.get(phrase_base, [])

                for specialty in specialties:
                    if specialty not in combined_results:
                        combined_results[specialty] = {
                            "total_diff": 0.0,
                            "count": 0
                        }

                    combined_results[specialty]["total_diff"] += prob_diff
                    combined_results[specialty]["count"] += 1
        count += 1
        if count >= 10:
            break

    # Compute averages
    for specialty in combined_results:
        count = combined_results[specialty]["count"]
        combined_results[specialty]["average_diff"] = (
            combined_results[specialty]["total_diff"] / count if count > 0 else 0.0
        )

    return combined_results, all_base_probs_to_save, all_adapt_probs_to_save


# Specialty filtering function (example: filter for cardiology terms only)
def specialty_filter_fn(specialties):
    return len(specialties) < 4 and "Others" not in specialties


if __name__ == "__main__":
    # Load tokenizer and model configuration
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load term dictionaries
    term_dict_paths = [
        "all_terms_Cardiology_test.json",
        "all_terms_Gastroenterology_test.json",
        "all_terms_Infectious_diseases_test.json",
        "all_terms_Neurology_test.json",
        "all_terms_Obstetrics_gynecology_test.json",
        "all_terms_Pediatrics_test.json"
    ]
    term_dicts = []
    for term_dict_path in term_dict_paths:
        with open(term_dict_path, "r") as f:
            term_dicts.append(json.load(f))

    # Datasets and adapters
    dataset_paths = [
        "Cardiology_idx_test.json",
        "Gastroenterology_idx_test.json",
        "Infectious_diseases_idx_test.json",
        "Neurology_idx_test.json",
        "Obstetrics_gynecology_idx_test.json",
        "Pediatrics_idx_test.json"
    ]

    adapter_list = ["list_of_adapters"]

    # Output files
    output_paths = ["list_of_output_files"]

    medqa_test_length = {
        "Cardiology": 80,
        "Gastroenterology": 83,
        "Infectious_diseases": 102,
        "Neurology": 74,
        "Obstetrics_gynecology": 88,
        "Pediatrics": 90
    }

    # Process datasets and adapters
    for dataset_path, term_dict, output_path in zip(dataset_paths, term_dicts, output_paths):

        for spe in list(medqa_test_length.keys()):
            if spe in dataset_path:
                number_to_keep = medqa_test_length[spe]
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
            dataset = dataset[:number_to_keep]
            random.shuffle(dataset)
        
        output_base_results = []
        output_adapt_results = []

        for adapter_path in adapter_list:
            results, all_base_probs, all_adapt_probs = process_dataset(dataset, adapter_path, model_name, tokenizer, term_dict, specialty_filter_fn)
            print(f"Results for Dataset: {dataset_path}, Adapter: {adapter_path}")
            print(json.dumps(results, indent=4))
            output_base_results.append(all_base_probs)
            output_adapt_results.append(all_adapt_probs)

        with open(output_path.replace(".json", "_filter_all_base.json"), "w") as f:
            json.dump(output_base_results, f, indent=4)
        
        with open(output_path.replace(".json", "_filter_all_adapt.json"), "w") as f:
            json.dump(output_adapt_results, f, indent=4)
