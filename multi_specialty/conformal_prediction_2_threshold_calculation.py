import pandas as pd
import numpy as np
import random
from collections import Counter
import re

list_of_specialties = ['Emergency medicine', 'Allergist', 'Anaesthetics', 'Cardiology', 'Child psychiatry', 'Clinical biology', 'Clinical chemistry', 'Clinical microbiology', 'Clinical neurophysiology', 'Craniofacial surgery', 'Dermatology', 'Endocrinology', 'Family and General Medicine', 'Gastroenterologic surgery', 'Gastroenterology', 'General Practice', 'General surgery', 'Geriatrics', 'Hematology', 'Immunology', 'Infectious diseases', 'Internal medicine', 'Laboratory medicine', 'Nephrology', 'Neuropsychiatry', 'Neurology', 'Neurosurgery', 'Nuclear medicine', 'Obstetrics and gynecology', 'Occupational medicine', 'Oncology', 'Ophthalmology', 'Oral and maxillofacial surgery', 'Orthopedics', 'Otorhinolaryngology', 'Pediatric surgery', 'Pediatrics', 'Pathology', 'Pharmacology', 'Physical medicine and rehabilitation', 'Plastic surgery', 'Podiatric surgery', 'Preventive medicine', 'Psychiatry', 'Public health', 'Radiation Oncology', 'Radiology', 'Respiratory medicine', 'Rheumatology', 'Stomatology', 'Thoracic surgery', 'Tropical medicine', 'Urology', 'Vascular surgery', 'Venereology', 'Others']
formatted_specialties = ["*" + s + "*" for s in list_of_specialties]
list_of_lower_specialties = [s.lower() for s in list_of_specialties]


# Step 1: Load labeled datasets
# Assuming the three CSV files have the same structure with "Question" and multiple label columns
file1 = "path_to_file_1"
file2 = "path_to_file_2"
file3 = "path_to_file_3"

# Load gold label files as dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

df_gold = pd.concat([df1, df2, df3], ignore_index=True)

label_columns_1 = ["Relavant1","Relavant2","ExtraLabel1","ExtraLabel2"]  # Columns for file1
label_columns_2 = ["Relavant1","Relavant2","ExtraLabel1","ExtraLabel2"]  # Columns for file2
label_columns_3 = ["MainLabel","Relavant1","Relavant2","Relavant3","ExtraLabel1","ExtraLabel2"]  # Columns for file3 (with extra column)

df_gold["Gold Labels"] = df_gold[label_columns_3].apply(lambda row: [str(x).lower() for x in row.dropna()], axis=1)

# Step 2: Randomly select 200 questions for the calibration set
random.seed(42)
calibration_indices = random.sample(range(len(df_gold)), 200)
print(f"Calibration indices: {calibration_indices}")

# Step 3: Load GPT-generated responses (assumed to already exist in a CSV file)
# The GPT results should have columns: "Question number", "Question", and "Response"
def preprocess_response(response, specialty_list):
    """
    Preprocess a single GPT response.
    - If the response (lowercase) matches a specialty name, keep it.
    - Otherwise, extract content between '*' or '**' and check if it matches a specialty name.
    """
    response = response.strip().lower()  # Standardize response to lowercase and strip whitespace

    # Check if the response is exactly a specialty name
    if response in specialty_list:
        print("Exact match found")
        return response
    
    found = False
    for sp_item in specialty_list:
        if sp_item.lower() in response.lower():
            print(f"Partial match found for {sp_item}")
            found = True
            return sp_item
    if not found:
        return "Others"

    # If no match is found, return None or an empty string
    return "Others"

gpt_results_file = "gpt_result_file"
df_gpt = pd.read_csv(gpt_results_file)
df_gpt["Processed Response"] = df_gpt["Response"].apply(lambda x: preprocess_response(x, list_of_lower_specialties))


# Step 4: Parse GPT responses and compute frequencies
non_conformity_scores = []

# Loop through calibration indices
for idx in calibration_indices:
    # Get the corresponding question and its gold labels
    print(f"Processing question: {idx}")
    question_row = df_gold.iloc[idx]
    print(f"Question: {question_row['Question']}")
    gold_labels = question_row["Gold Labels"]  # List of correct labels for the question
    
    # Get the GPT responses for this question (20 rows)
    gpt_responses = df_gpt[df_gpt["Question"] == question_row["Question"]]["Processed Response"]
    
    # Count the frequency of each response
    response_frequencies = Counter(gpt_responses)
    total_responses = sum(response_frequencies.values())
    
    # Normalize frequencies to get probabilities
    probabilities = {label: count / total_responses for label, count in response_frequencies.items()}
    
    # Compute P(Ci) for the correct answers
    label_probabilities = []
    for label in gold_labels:
        if label in probabilities:
            label_probabilities.append(probabilities[label])
    if len(label_probabilities) == 0:
        min_probability = 0.0
    else:
        min_probability = min(label_probabilities)
    print(f"min_probability: {min_probability}")
    
    # Compute non-conformity score Si = 1 - P(Ci)
    non_conformity_score = 1 - min_probability
    non_conformity_scores.append(non_conformity_score)

# Step 5: Compute the (1-alpha) quantile for calibration threshold
alpha = 0.1
print(f"Non-conformity scores: {non_conformity_scores}")
calibration_threshold = np.quantile(non_conformity_scores, 1 - alpha)

# Step 6: Save results to a CSV file
output_file = "output_path"
output_data = {
    "Question number": calibration_indices,
    "Question": df_gold.iloc[calibration_indices]["Question"],
    "Non-conformity score": non_conformity_scores
}
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file, index=False)

# Print the threshold
print(f"Calibration threshold (1-alpha quantile with alpha={alpha}): {calibration_threshold}")