import numpy as np
from openai import OpenAI
import pandas as pd

api_key = "Your API key"
client = OpenAI(api_key=api_key)

# Function to check if the first output matches a phrase in vocab
def match_output_with_vocab(output_text, vocab):
    """
    Matches the first word or phrase in the output text with the vocab.

    Args:
        output_text (str): The model's output text.
        vocab (list): List of valid phrases to match.

    Returns:
        str or None: The matched phrase from the vocab, or None if no match.
    """
    # Check if any phrase in vocab matches the start of the output text
    for phrase in vocab:
        if output_text.lower().startswith(phrase.lower()):  # Case-insensitive match
            return phrase
    return None


def query_gpt_with_logprobs(question, vocab, model="gpt-3.5-turbo", max_tokens=50, default_prob=1e-6):
    """
    Queries GPT, extracts log probabilities, and matches vocabulary phrases.

    Args:
        prompt (str): The user input prompt.
        vocab (list): List of vocabulary phrases to match.
        model (str): The OpenAI model to use.
        max_tokens (int): Maximum tokens for GPT response.
        default_prob (float): Probability assigned to unmatched vocab phrases.

    Returns:
        list of tuples: [(phrase, probability)]
    """
    # Query GPT with logprobs
    prompt = f"""Please classify the medical multiple choice question into ONE OR MORE of the following clinical specialties according to the decending order of clinical relevance: " + ", ".join(formatted_specialties) + ". Mention ALL relevant specialties and do not mention any irrelevant specialties.

    User: {question}
    Assistant: """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=3,
        temperature=0
    )

    # Extract logprobs from response
    logprobs_list = response.choices[0].logprobs.content  # List of ChatCompletionTokenLogprob objects
    gpt_output_text = response.choices[0].message.content
    print(f"GPT output text: {gpt_output_text}")
    
    # Store tokenized output in order
    token_list = []  # Ordered list of tokens
    token_probs = {}  # Dictionary {token: probability}
    
    for logprob_data in logprobs_list:
        token = logprob_data.token
        logprob = logprob_data.logprob
        probability = np.exp(logprob)  # Convert log prob to normal prob
        
        token_list.append(token)  # Maintain order
        token_probs[token] = probability  # Store probability
    # print(f"token_list: {token_list}")
    # print(f"token_probabilities: {token_probs}")
    tokenized_text = "".join(token_list).replace(" ", "").lower()
    # print(f"tokenized text: {tokenized_text}")

    # Sort vocab by length (longer phrases first) to prioritize precise matches
    vocab = sorted(vocab, key=lambda x: -len(x.split()))
    
    # Match vocabulary against GPT output
    phrase_probabilities = {}

    for phrase in vocab:
        phrase_no_spaces = phrase.replace(" ", "").lower()  # Remove spaces for matching
        phrase_words = phrase.lower().split()  # Get list of words in phrase
        matched_probability = default_prob  # Default probability if no match is found

        # Exact Match Check
        if phrase_no_spaces in tokenized_text:
            print(f"Exact Match: '{phrase}' in Tokenized Output")

            # Extract probabilities of matching tokens
            matched_probs = []
            start_index = tokenized_text.index(phrase_no_spaces)
            end_index = start_index + len(phrase_no_spaces)

            current_pos = 0
            for token in token_list:
                token_length = len(token)
                if current_pos >= start_index and current_pos + token_length <= end_index:
                    matched_probs.append(token_probs[token])
                current_pos += token_length

            matched_probability = np.mean(matched_probs) if matched_probs else default_prob

        else:
            # Partial Match Check
            for i in range(1, len(phrase_words) + 1):
                partial_phrase = "".join(phrase_words[:i])  # Take first i words
                if partial_phrase in tokenized_text:
                    print(f"⚠️ Partial Match: '{phrase}' → Found '{' '.join(phrase_words[:i])}'")

                    # Extract probabilities of matching tokens
                    matched_probs = []
                    start_index = tokenized_text.index(partial_phrase)
                    end_index = start_index + len(partial_phrase)

                    current_pos = 0
                    for token in token_list:
                        token_length = len(token)
                        if current_pos >= start_index and current_pos + token_length <= end_index:
                            matched_probs.append(token_probs[token])
                        current_pos += token_length

                    partial_match_score = i / len(phrase_words)  # Score based on words matched
                    matched_probability = (np.mean(matched_probs) if matched_probs else default_prob) * partial_match_score
                    break  # Stop at first partial match

        # Store the probability for the vocabulary phrase
        phrase_probabilities[phrase] = matched_probability

    return phrase_probabilities  # Return phrase probabilities



# Function to evaluate conformal prediction
def evaluate_conformal_prediction(calibration_dataset, vocab, epsilon):
    """
    Evaluates conformal prediction based on calibration data.

    Args:
        calibration_dataset (list): List of (question, true_labels) tuples.
        vocab (list): List of valid phrases to match.
        epsilon (float): Conformal prediction threshold.

    Returns:
        list: Calibration scores for the dataset.
    """
    scores = []  # Calibration scores
    covered = 0  # Count of covered questions
    total_questions = len(calibration_dataset)
    y_true = []  # True labels for all samples
    y_pred = []  # Predicted labels for all samples

    for question, true_labels in calibration_dataset:
        # Query GPT and get probabilities for the matched phrases
        matched_probs = query_gpt_with_logprobs(question, vocab)  # Returns {phrase: probability}

        # Ensure every phrase in vocab is in matched_probs (default to small probability)
        matched_probs = {phrase: matched_probs.get(phrase, 0.01) for phrase in vocab}

        # Get the highest probability among the true labels
        true_probs = [matched_probs[label] for label in true_labels if label in matched_probs]

        if true_probs: 
            max_prob = max(true_probs)  # Use the highest probability for calibration
        else:
            max_prob = 0.01  # Default to small probability if no match

        # Compute calibration score (1 - max_prob is a simple formulation)
        scores.append(1 - max_prob)

        # Prediction set: all labels with probabilities above the threshold
        prediction_set = [label for label, prob in matched_probs.items() if (1 - prob) <= np.percentile(scores, (1 - epsilon) * 100)]

        # Update y_true and y_pred for F1 score calculation
        y_true.append(set(true_labels))
        y_pred.append(set(prediction_set))

        # Check if true_labels are covered by prediction_set
        if set(true_labels).intersection(prediction_set):
            covered += 1
        
        # Step 2: Compute the Conformal Threshold (1 - ε percentile)
    threshold = np.percentile(scores, (1 - epsilon) * 100)
    print(f"Conformal Threshold (ε={epsilon}): {threshold:.4f}")

    # Compute coverage
    coverage = covered / total_questions

    # Compute precision, recall, and F1 score
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for true, pred in zip(y_true, y_pred):
        tp = len(true & pred)  # True positives
        fp = len(pred - true)  # False positives
        fn = len(true - pred)  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1_score

    # Average precision, recall, and F1 score
    precision = total_precision / total_questions
    recall = total_recall / total_questions
    f1_score = total_f1 / total_questions

    return {
        "scores": scores,
        "threshold": threshold,
        "coverage": coverage,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def apply_conformal_prediction(test_dataset, vocab, calibration_scores, epsilon):
    """
    Applies conformal prediction to the test dataset using the calibration scores.

    Args:
        test_dataset (list): List of (question, true_labels) tuples.
        vocab (list): List of valid phrases to match.
        calibration_scores (list): Calibration scores from the calibration dataset.
        epsilon (float): Conformal prediction threshold.

    Returns:
        float: Coverage of the test dataset under the conformal prediction threshold.
    """
    # Calculate the conformal threshold
    threshold = np.percentile(calibration_scores, (1 - epsilon) * 100)

    covered = 0  # Count of covered questions
    total_questions = len(test_dataset)

    for question, true_labels in test_dataset:
        # Query GPT and get probabilities for the matched phrase
        matched_probs = query_gpt_with_logprobs(question, vocab)

        # Get the first tokens of the correct labels
        first_tokens = {label.split()[0] for label in true_labels}

        # Calculate the largest probability of the correct tokens
        max_prob = max(matched_probs.get(token, 0) for token in first_tokens)

        # Check if the prediction is within the conformal threshold
        score = 1 - max_prob
        if score <= threshold:  # Valid prediction
            covered += 1

    # Compute coverage
    coverage = covered / total_questions
    print(f"Coverage: {coverage * 100:.2f}%")
    return coverage


def process_calibration_csv_pandas(file_path):
    """
    Process a CSV file where the first column contains questions and subsequent columns contain true labels.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        A list of tuples [(question, true_labels), ...], where:
            - question is a string
            - true_labels is a list of strings (words/phrases)
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Process the data row by row
    calibration_dataset = []
    for _, row in df.iterrows():
        question = row.iloc[0].strip()  # The first column is the question
        true_labels = [str(label).strip() for label in row.iloc[1:] if pd.notnull(label)]  # Remaining columns
        calibration_dataset.append((question, true_labels))
    
    return calibration_dataset


# Example Usage
if __name__ == "__main__":
    # Define a calibration dataset (question, true_labels)
    calibration_file = "/home/xyan1/srp_project/multi_specialty/Annotator_2_done.csv"
    calibration_dataset = process_calibration_csv_pandas(calibration_file)
    for path in ["/home/xyan1/srp_project/multi_specialty/Annotator_3_done.csv", "/home/xyan1/srp_project/multi_specialty/Annotator_4_done.csv"]:
        calibration_file = process_calibration_csv_pandas(path)
        calibration_dataset.extend(calibration_file)

    # Vocabulary containing only the first tokens of the true labels
    vocab = {'Emergency medicine', 'Allergist', 'Anaesthetics', 'Cardiology', 'Child psychiatry', 'Clinical biology', 'Clinical chemistry', 'Clinical microbiology', 'Clinical neurophysiology', 'Craniofacial surgery', 'Dermatology', 'Endocrinology', 'Family and General Medicine', 'Gastroenterologic surgery', 'Gastroenterology', 'General Practice', 'General surgery', 'Geriatrics', 'Hematology', 'Immunology', 'Infectious diseases', 'Internal medicine', 'Laboratory medicine', 'Nephrology', 'Neuropsychiatry', 'Neurology', 'Neurosurgery', 'Nuclear medicine', 'Obstetrics and gynecology', 'Occupational medicine', 'Oncology', 'Ophthalmology', 'Oral and maxillofacial surgery', 'Orthopedics', 'Otorhinolaryngology', 'Pediatrics', 'Pathology', 'Pharmacology', 'Physical medicine and rehabilitation', 'Plastic surgery', 'Podiatric surgery', 'Preventive medicine', 'Psychiatry', 'Public health', 'Radiation Oncology', 'Radiology', 'Respiratory medicine', 'Rheumatology', 'Stomatology', 'Thoracic surgery', 'Tropical medicine', 'Urology', 'Vascular surgery', 'Venereology', 'Others'}

    # Conformal prediction threshold
    epsilon = 0.05

    # Evaluate conformal prediction
    calibration_scores = evaluate_conformal_prediction(calibration_dataset, vocab, epsilon)
    print("Calibration Scores:", calibration_scores)
