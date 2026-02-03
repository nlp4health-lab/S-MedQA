# S-MedQA Dataset

Welcome to the S-MedQA dataset repository! S-MedQA is the first clinical specialty annotated medical multiple-choice question-answer (QA) dataset. This repository contains the S-MedQA dataset, along with documentation and code to facilitate its use in medical QA research.

## Overview

S-MedQA is a high-quality benchmark for medical QA with clinical specialty annotations. The dataset is sourced from MedQA and annotated with clinical specialties using GPT-3.5. We provide multiple versions of the dataset, allowing users to select between different accuracy and coverage trade-offs.

## Dataset Versions

We release multiple versions of S-MedQA with different accuracy/coverage trade-offs. Users can choose a cleaner version of the dataset with fewer examples or a more noisy version with more examples. The different versions are created using varying thresholds for majority voting to include an example in the final dataset.

### Accuracy vs. Coverage

In Table 1 below, we present the trade-off between accuracy and coverage based on the minimum number of votes in agreement required for an example to be included in the dataset.

| Minimum Votes | Accuracy (%) | Coverage (%) |
|---------------|--------------|--------------|
| 1             | 72.8 - 80.2  | 89.1         |
| 3             | 90.8         | 49.2         |
| 5             | 97.8         | 49.2         |

We select a quorum of 3 votes for the default version of the dataset in this study, providing an adequate balance of fine-tuning data.

## Development Process

### Medical Specialty Categorization

We sourced examples from MedQA, a commonly used dataset for evaluating medical LLMs. Using GPT-3.5, we annotated samples with clinical specialties. Initially, we observed low accuracy (~75%) using a single prompt. To improve accuracy, we designed five prompts and applied majority voting to generate predictions with GPT-3.5 for each sample.

### Dataset Splits

After annotation, we excluded 1,324 samples with no majority vote and 308 samples categorized as "Others" for containing clinically irrelevant information. We retained 15 out of 55 specialties with more than 200 samples each. The final dataset comprises 7,125 / 899 / 893 samples in train/validation/test sets.

### Manual Validation

A medical expert labeled each example in the validation and test sets and validated 1,000 random samples from the train set. The accuracy of our annotations improved significantly with multiple prompts and voting (from 72.8%-80.2% to 90.8%-97.8%). To ensure the expert's trustworthiness, we achieved an inter-annotator agreement of 83.6% (95% CI [69.0%, 93.9%]) among the expert and three medical master students.

## Usage

### Choosing the Version

We provide individual categorizations and votes for all examples, allowing users to decide their preference between accuracy and coverage based on their specific use cases.

### Loading the Dataset

To load the dataset into your project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/s-medqa.git
    ```

2. Navigate to the dataset directory:
    ```bash
    cd s-medqa
    ```

3. Load the dataset in your script:
    ```python
    import json

    # Load the train, validation, and test sets
    train_df = json.load('data/S-MedQA_train.json')
    val_df = json.load('data/S-MedQA_validation.json')
    test_df = json.load('data/S-MedQA_test.json')
    ```

