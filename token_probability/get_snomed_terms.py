import requests
import json
from scttsrapy.api import EndpointBuilder
from scttsrapy.concepts import find_concepts_term
from scttsrapy.concepts import get_concept_ancestors
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from term_extraction import extract_terms_from_dataset
import spacy
import scispacy
import en_core_sci_sm
from scispacy.linking import EntityLinker

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

endpoint_builder = EndpointBuilder()
endpoint_builder.set_api_endpoint("endpoint_url")

print("start processing scispacy")
nlp = en_core_sci_sm.load()
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")
print("scispacy successfully loaded!")

disease_to_specialty = {
    "49601007": "Cardiology", # Disorder of cardiovascular system (disorder)
    "53619000": "Gastroenterology", # Disorder of digestive system (disorder)
    "40733004": "Infectious diseases", # Infectious disease (disorder)
    "363124003": "Obstetrics and gynecology", # Disorder of female reproductive system (disorder)
    "1269083008": "Obstetrics and gynecology", # isorder of fetus and/or mother during labor
    "118940003": "Neurology", # Disorder of nervous system (disorder)
    "231538003": "Pediatrics", # Behavioral and emotional disorder with onset in childhood (disorder)
    "414025005": "Pediatrics", # Disorder of fetus and/or newborn (disorder)
    "5294002": "Pediatrics" # Developmental disorder (disorder)
}

specialty_keywords = {
    "Cardiology": [
        "cardiovascular", "cardiac", "coronary", "heart", "myocardial", "arterial", 
        "vein", "vascular", "arrhythmia", "hypertension", "ischemia", "atherosclerosis",
        "angina", "tachycardia", "bradycardia", "pericarditis", "endocarditis", "cardiomyopathy", 
        "valvular", "atrial", "ventricular"
    ],
    "Gastroenterology": [
        "gastrointestinal", "stomach", "intestine", "colon", "rectum", "liver", 
        "esophagus", "bowel", "pancreas", "biliary", "gallbladder", "duodenum", 
        "hepatitis", "diarrhea", "constipation", "crohn", "colitis", "gastritis",
        "reflux", "ulcer", "digestive", "celiac", "nausea", "vomiting", "IBD", "IBS"
    ],
    "Infectious diseases": [
        "infection", "bacteria", "virus", "fungus", "parasitic", "sepsis", "fever", 
        "antibiotic", "antiviral", "pathogen", "microbial", "epidemic", "pandemic", 
        "contagion", "pneumonia", "tuberculosis", "HIV", "AIDS", "malaria", 
        "meningitis", "hepatitis", "measles", "dengue", "typhoid", "COVID", "influenza", 
        "fungal", "parasitic", "zoonotic", "tetanus", "rabies", "lyme", "syphilis", "leprosy"
    ],
    "Neurology": [
        "nervous system", "brain", "spinal cord", "peripheral nerves", "neuron", 
        "headache", "migraine", "stroke", "seizure", "epilepsy", "parkinson", 
        "dementia", "alzheimer", "multiple sclerosis", "neuropathy", "myopathy", 
        "cerebral", "neurodegenerative", "neuroinflammatory", 
        "motor neuron", "ataxia", "tremor", "neurodevelopmental", "autism", 
        "ADHD", "ALS", "cerebral palsy", "encephalitis", "myelitis"
    ],
    "Obstetrics and gynecology": [
        "pregnancy", "labor", "delivery", "menstrual", "fertility", "infertility", 
        "ovary", "uterus", "cervix", "vagina", "endometrial", "placenta", 
        "fetus", "newborn", "contraception", "menopause", "gynecological", 
        "postpartum", "prenatal", "miscarriage", "abortion", "hysterectomy", 
        "cesarean", "PCOS", "fibroid", "endometriosis", "ectopic", "mastitis", 
        "breastfeeding", "cervical cancer", "pap smear", "HPV", "IVF"
    ],
    "Pediatrics": [
        "child", "infant", "newborn", "toddler", "adolescent", "neonatal", 
        "growth", "development", "vaccination", "immunization", "behavioral", 
        "autism", "ADHD", "learning disability", "pediatric asthma", "bronchiolitis", 
        "pediatric diabetes", "congenital", "birth defect", "pediatric cancer", 
        "juvenile arthritis", "rickets", "malnutrition", "failure to thrive", 
        "measles", "mumps", "rubella", "whooping cough", "chickenpox", 
        "pediatric cardiology", "developmental disorder", "speech delay"
    ]
}

def map_term_to_specialty(term, disease_to_specialty, endpoint_builder):
    """
    Maps a term to one or more specialties based on SNOMED CT and HPO data.
    """
    # Initialize the result dictionary for the given term
    map_result = {term: set()}  # Use a set to avoid duplicated specialties

    # Step 1: Find the corresponding SNOMED CT term
    result = find_concepts_term(term=term, endpoint_builder=endpoint_builder)

    if not result["success"]:
        print(f"Failed to map UMLS concept to SNOMED CT for term: {term}.")
        map_result[term] = ["Others"]  # If it fails, set as "Others"
        return map_result

    # Check for the first active SNOMED CT concept
    active_concept = None
    for concept in result["content"]["items"]:
        if concept.get("active") == True:  # Ensure the concept is active
            active_concept = concept
            break

    if not active_concept:
        print(f"No active SNOMED CT concept found for term: {term}.")
        map_result[term] = ["Others"]
        return map_result

    # Get the active concept's ID and FSN (fully specified name)
    snomed_concept_id = active_concept["conceptId"]
    snomed_fsn = active_concept["fsn"]["term"]

    # Step 2: Check the type of concept
    if any(category in snomed_fsn for category in ["disorder", "finding", "observable entity", "procedure", "body structure"]):
        # Step 2.1: If the concept is a disorder
        if "disorder" in snomed_fsn:
            # Map disorder to specialties
            specialties = map_disorder_to_specialty(snomed_concept_id, disease_to_specialty, endpoint_builder)
            if not specialties:
                map_result[term].add("Others")  # If no specialty found, add "Others"
            else:
                map_result[term].update(specialties)  # Add specialties and deduplicate

        # Step 2.2: For findings, procedures, observable entities, and body structures
        elif any(category in snomed_fsn for category in ["finding", "observable entity", "procedure", "body structure"]):
            # Step 2.2.1: Check if specialty keywords exist in ancestors
            ancestor_result = get_concept_ancestors(concept_id=snomed_concept_id, form="inferred", endpoint_builder=endpoint_builder)
            if ancestor_result["success"]:
                ancestors = ancestor_result["content"]
                mapped_to_specialty = False  # Flag to track if specialty was directly mapped

                # Iterate over ancestors and check for specialty keywords
                for ancestor in ancestors:
                    ancestor_fsn = ancestor["fsn"]["term"].lower()  # Convert FSN to lowercase for case-insensitive matching
                    for specialty, keywords in specialty_keywords.items():
                        if any(keyword in ancestor_fsn for keyword in keywords):
                            print(f"Specialty keyword '{keywords}' found in ancestor '{ancestor_fsn}'. Mapping to '{specialty}'.")
                            map_result[term].add(specialty)
                            mapped_to_specialty = True
                            break
                    if mapped_to_specialty:
                        break  # Stop checking further ancestors if a match is found

                # Step 2.2.2: If no specialty keywords found, map related disorders
                if not mapped_to_specialty:
                    related_disorders = [ancestor for ancestor in ancestors if "disorder" in ancestor["fsn"]["term"]]

                    if related_disorders:
                        # Map related disorders to specialties
                        for disorder in related_disorders:
                            specialties = map_disorder_to_specialty(disorder["conceptId"], disease_to_specialty, endpoint_builder)
                            map_result[term].update(specialties)  # Add specialties and deduplicate

                        if not map_result[term]:  # If no specialties found, classify as "Others"
                            map_result[term].add("Others")
                    else:
                        # If no related disorders, fallback to HPO search
                        hpo_specialties = map_finding_via_hpo(term, disease_to_specialty, endpoint_builder)
                        map_result[term].update(hpo_specialties)

            else:
                print(f"Failed to retrieve ancestors for term: {term}.")
                map_result[term].add("Others")

    else:
        print(f"Term '{term}' is neither a disorder, finding, observable entity, nor procedure.")
        map_result[term].add("Others")

    # Convert the set back to a list for the final result
    map_result[term] = list(map_result[term])
    return map_result


def map_disorder_to_specialty(disorder_concept_id, disease_to_specialty, endpoint_builder):
    """
    Maps a disorder to its high-level specialty based on its ancestors.
    """
    ancestor_result = get_concept_ancestors(concept_id=disorder_concept_id, form="inferred", endpoint_builder=endpoint_builder)
    if not ancestor_result["success"]:
        print(f"Failed to retrieve ancestors for disorder: {disorder_concept_id}.")
        return []

    # Check ancestors for high-level classes that map to specialties
    ancestors = ancestor_result["content"]
    specialties = set()  # Use a set to avoid duplicates
    for ancestor in ancestors:
        if ancestor["conceptId"] in disease_to_specialty:
            specialties.add(disease_to_specialty[ancestor["conceptId"]])  # Add specialty

    return list(specialties)  # Return as a list


def map_finding_via_hpo(original_term, disease_to_specialty, endpoint_builder):
    """
    Attempts to map a finding (symptom) to specialties using HPO data.
    If no specialties are found for the original term, iteratively simplify the term
    (removing adjectives) and search again until a specialty is found or no adjectives remain.
    """
    # Load HPO data
    try:
        with open("hp.json", 'r') as file:
            hpo_data = json.load(file)
    except Exception as e:
        print(f"Failed to load HPO file: {e}")
        return ["Others"]

    # Start with the original term
    current_term = original_term
    specialties = []

    while True:
        # Step 1: Search for the current term in HPO
        print(f"Searching HPO for term: '{current_term}'")
        matching_labels = search_hpo_for_term(current_term.lower(), hpo_data)

        if matching_labels:
            print(f"Matches found in HPO for term '{current_term}': {matching_labels}")

            # Step 2: Map HPO labels back to SNOMED CT and find specialties
            for label in matching_labels:
                term_map_result = find_concepts_term(term=label, endpoint_builder=endpoint_builder)
                if term_map_result["success"]:
                    mapped_terms = term_map_result["content"]["items"]
                    for mapped_term in mapped_terms:
                        if "disorder" in mapped_term["fsn"]["term"]:
                            # Map the disorder to specialties
                            specialties.extend(
                                map_disorder_to_specialty(mapped_term["conceptId"], disease_to_specialty, endpoint_builder)
                            )

            # If specialties are found, stop the loop and return them
            if specialties:
                return specialties

        # Step 3: Simplify the term if no specialties are found
        simplified_term = simplify_term_using_nltk(current_term)
        if simplified_term == current_term:
            # Stop iterating if no more adjectives can be removed
            print(f"No more adjectives to remove. Term '{current_term}' mapped to 'Others'.")
            return ["Others"]

        print(f"No specialties found for term '{current_term}'. Retrying with simplified term: '{simplified_term}'")
        current_term = simplified_term


def search_hpo_for_term(term, hpo_data):
    """
    Searches for a term in the HPO data and returns matching labels.
    """
    print(f"Searching HPO for term: '{term}'")
    matching_labels = []
    for graph in hpo_data.get("graphs", []):
        for node in graph.get("nodes", []):
            # Check if the term appears in any of the node's values
            if any(term.lower() in str(value).lower() for value in node.values()):
                lbl_value = node.get("lbl")
                if lbl_value:
                    matching_labels.append(lbl_value)

    print(f"Matches found for term '{term}': {matching_labels}")
    return matching_labels


def simplify_term_using_nltk(term):
    """
    Simplifies a term by removing adjectives (JJ) from the beginning of the phrase.
    """
    tokens = word_tokenize(term)
    tagged_tokens = pos_tag(tokens)

    simplified_tokens = []
    adjective_found = True

    for word, tag in tagged_tokens:
        if tag != "JJ" or not adjective_found:
            # Add the word if it's not an adjective or we have already passed adjectives
            simplified_tokens.append(word)
            adjective_found = False  # Stop filtering adjectives after the first non-adjective is found

    if len(simplified_tokens) != 0:
        simplified_term = " ".join(simplified_tokens)
        print(f"Simplified term '{term}' to '{simplified_term}'.")
    else:
        simplified_term = term
    
    return simplified_term


# Create a mapping

def process_multiple_datasets_with_cache(dataset_paths, disease_to_specialty, endpoint_builder, output_file):
    """
    Processes multiple datasets to extract terms, map them to specialties,
    and save the results in a single JSON file, avoiding duplicate mapping.
    
    Parameters:
    - dataset_paths (list): List of paths to dataset JSON files.
    - disease_to_specialty (dict): Mapping of SNOMED CT disorder concept IDs to medical specialties.
    - endpoint_builder (EndpointBuilder): Configured endpoint for SNOMED CT API.
    - output_file (str): Path to the output JSON file to save results.

    Returns:
    - None: Results are saved to the specified output file.
    """
    all_results = {}
    cache = {} 

    for dataset_path in dataset_paths:
        try:
            print(f"Processing dataset: {dataset_path}")
            
            # Extract terms from the current dataset
            terms = extract_terms_from_dataset(dataset_path, nlp, linker)
            if not terms:
                print(f"No terms found in dataset: {dataset_path}")
                continue
            
            print(f"Extracted terms: {terms}")

            for original_term, umls_term in terms.items():
                if original_term in cache:
                    # Use cached result if original term is already mapped
                    print(f"Original term '{original_term}' already mapped. Using cached result.")
                    all_results[original_term] = cache[original_term]
                else:
                    # if tui not in ["T098", "T079", "T109", "T081", "T170", "T101", "T052", "T083", ""]
                    print(f"Mapping term '{original_term}' using UMLS term '{umls_term}'...")
                    # Map the term to specialties using the UMLS term
                    term_specialty_mapping = map_term_to_specialty(umls_term, disease_to_specialty, endpoint_builder)
                    mapped_specialties = term_specialty_mapping[umls_term]
                    
                    # Cache the result with the original term
                    cache[original_term] = mapped_specialties
                    all_results[original_term] = mapped_specialties

        except Exception as e:
            print(f"An error occurred while processing dataset {dataset_path}: {e}")

    # Save all results to the specified output file
    try:
        with open(output_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)
        print(f"Results successfully saved to: {output_file}")
    except Exception as e:
        print(f"An error occurred while saving results to {output_file}: {e}")


# Example usage
dataset_paths = [
    "medqa_six_train.json",
    "..."
]
output_file = "term_specialty_mapping_train.json"

# Call the function
process_multiple_datasets_with_cache(dataset_paths, disease_to_specialty, endpoint_builder, output_file)