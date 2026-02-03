# import spacy
# import scispacy
# import en_core_sci_sm
# from scispacy.linking import EntityLinker
import json
import re

# nlp = en_core_sci_sm.load()

# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
# linker = nlp.get_pipe("scispacy_linker")

# # Load the ScispaCy model and configure the linker
# nlp = en_core_sci_sm.load()
# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
# linker = nlp.get_pipe("scispacy_linker")

def extract_terms_from_dataset(dataset_path, nlp, linker):
    """
    Extracts medical terms from a dataset using ScispaCy and UMLS linking.

    Args:
        dataset (list[dict]): A list of entries, where each entry is a dictionary
                              containing 'instruction' and 'input' keys.

    Returns:
        list: A list of unique medical terms extracted from the dataset.
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    terms = {}

    count = 0
    for entry in dataset:
        # Combine 'instruction' and 'input' into a single sentence
        options = re.sub(r'[A-D]\.\s*', '', entry['input']).replace('\n', '. ').strip()
        sentence = entry['instruction'] + " " + options

        # Process the sentence with ScispaCy
        doc = nlp(sentence)

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
                        terms[entity.text] = entity_name
                        print(terms)
    return terms
