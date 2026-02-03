import json
from openai import OpenAI
from collections import Counter

with open("path_to_input_data", "r", encoding="utf-8") as f:
    sampled_data = json.load(f)


list_of_specialties = ['Emergency medicine', 'Allergist', 'Anaesthetics', 'Cardiology', 'Child psychiatry', 'Clinical biology', 'Clinical chemistry', 'Clinical microbiology', 'Clinical neurophysiology', 'Craniofacial surgery', 'Dermatology', 'Endocrinology', 'Family and General Medicine', 'Gastroenterologic surgery', 'Gastroenterology', 'General Practice', 'General surgery', 'Geriatrics', 'Hematology', 'Immunology', 'Infectious diseases', 'Internal medicine', 'Laboratory medicine', 'Nephrology', 'Neuropsychiatry', 'Neurology', 'Neurosurgery', 'Nuclear medicine', 'Obstetrics and gynecology', 'Occupational medicine', 'Oncology', 'Ophthalmology', 'Oral and maxillofacial surgery', 'Orthopedics', 'Otorhinolaryngology', 'Pediatric surgery', 'Pediatrics', 'Pathology', 'Pharmacology', 'Physical medicine and rehabilitation', 'Plastic surgery', 'Podiatric surgery', 'Preventive medicine', 'Psychiatry', 'Public health', 'Radiation Oncology', 'Radiology', 'Respiratory medicine', 'Rheumatology', 'Stomatology', 'Thoracic surgery', 'Tropical medicine', 'Urology', 'Vascular surgery', 'Venereology', 'Others']
formatted_specialties = ["*" + s + "*" for s in list_of_specialties]
list_of_lower_specialties = [s.lower() for s in list_of_specialties]

# normal
prompt_1 = "Please classify the medical multiple choice question into one of the following clinical specialties: " + ", ".join(formatted_specialties)
# explanation
prompt_2 = "Please classify the medical multiple choice question into one of the following clinical specialties and give exaplanations: " + ", ".join(formatted_specialties)
# normal + example
prompt_3 = "Please classify the medical multiple choice question into one of the following medical specialties: " + ", ".join(formatted_specialties)\
           + """\nExamples:
Question: A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5\u00b0C (97.7\u00b0F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?
Answer: Cardiology
Question: A 68-year-old man comes to the physician because of recurrent episodes of nausea and abdominal discomfort for the past 4 months. The discomfort is located in the upper abdomen and sometimes occurs after eating, especially after a big meal. He has tried to go for a walk after dinner to help with digestion, but his complaints have only increased. For the past 3 weeks he has also had symptoms while climbing the stairs to his apartment. He has type 2 diabetes mellitus, hypertension, and stage 2 peripheral arterial disease. He has smoked one pack of cigarettes daily for the past 45 years. He drinks one to two beers daily and occasionally more on weekends. His current medications include metformin, enalapril, and aspirin. He is 168 cm (5 ft 6 in) tall and weighs 126 kg (278 lb); BMI is 45 kg/m2. His temperature is 36.4\u00b0C (97.5\u00b0F), pulse is 78/min, and blood pressure is 148/86 mm Hg. On physical examination, the abdomen is soft and nontender with no organomegaly. Foot pulses are absent bilaterally. An ECG shows no abnormalities. Which of the following is the most appropriate next step in diagnosis?
Answer: Gastroenterology
Question: A 6-year-old male who recently immigrated to the United States from Asia is admitted to the hospital with dyspnea. Physical exam reveals a gray pseudomembrane in the patient's oropharynx along with lymphadenopathy. The patient develops myocarditis and expires on hospital day 5. Which of the following would have prevented this patient's presentation and decline?
Answer: Infectious diseases
Question: A 35-year-old woman with a history of Crohn disease presents for a follow-up appointment. She says that lately, she has started to notice difficulty walking. She says that some of her friends have joked that she appears to be walking as if she was drunk. Past medical history is significant for Crohn disease diagnosed 2 years ago, managed with natalizumab for the past year because her intestinal symptoms have become severe and unresponsive to other therapies. On physical examination, there is gait and limb ataxia present. Strength is 4/5 in the right upper limb. A T1/T2 MRI of the brain is ordered and is shown. Which of the following is the most likely diagnosis?
Answer: Neurology
Question: A 23-year-old G1 at 10 weeks gestation based on her last menstrual period is brought to the emergency department by her husband due to sudden vaginal bleeding. She says that she has mild lower abdominal cramps and is feeling dizzy and weak. Her blood pressure is 100/60 mm Hg, the pulse is 100/min, and the respiration rate is 15/min. She says that she has had light spotting over the last 3 days, but today the bleeding increased markedly and she also noticed the passage of clots. She says that she has changed three pads since the morning. She has also noticed that the nausea she was experiencing over the past few days has subsided. The physician examines her and notes that the cervical os is open and blood is pooling in the vagina. Products of conception can be visualized in the os. The patient is prepared for a suction curettage. Which of the following is the most likely cause for the pregnancy loss?
Answer: Obstetrics and gynecology
Question: An 8-month-old boy is brought to a medical office by his mother. The mother states that the boy has been very fussy and has not been feeding recently. The mother thinks the baby has been gaining weight despite not feeding well. The boy was delivered vaginally at 39 weeks gestation without complications. On physical examination, the boy is noted to be crying in his mother\u2019s arms. There is no evidence of cyanosis, and the cardiac examination is within normal limits. The crying intensifies when the abdomen is palpated. The abdomen is distended with tympany in the left lower quadrant. You suspect a condition caused by the failure of specialized cells to migrate. What is the most likely diagnosis?
Answer: Pediatrics\n
           """
# example + list of specialty options at last
prompt_4 = "Please classify the medical multiple choice question into one of the medical specialties."\
           + """\nExamples:
Question: A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5\u00b0C (97.7\u00b0F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?
Answer: Cardiology
Question: A 68-year-old man comes to the physician because of recurrent episodes of nausea and abdominal discomfort for the past 4 months. The discomfort is located in the upper abdomen and sometimes occurs after eating, especially after a big meal. He has tried to go for a walk after dinner to help with digestion, but his complaints have only increased. For the past 3 weeks he has also had symptoms while climbing the stairs to his apartment. He has type 2 diabetes mellitus, hypertension, and stage 2 peripheral arterial disease. He has smoked one pack of cigarettes daily for the past 45 years. He drinks one to two beers daily and occasionally more on weekends. His current medications include metformin, enalapril, and aspirin. He is 168 cm (5 ft 6 in) tall and weighs 126 kg (278 lb); BMI is 45 kg/m2. His temperature is 36.4\u00b0C (97.5\u00b0F), pulse is 78/min, and blood pressure is 148/86 mm Hg. On physical examination, the abdomen is soft and nontender with no organomegaly. Foot pulses are absent bilaterally. An ECG shows no abnormalities. Which of the following is the most appropriate next step in diagnosis?
Answer: Gastroenterology
Question: A 6-year-old male who recently immigrated to the United States from Asia is admitted to the hospital with dyspnea. Physical exam reveals a gray pseudomembrane in the patient's oropharynx along with lymphadenopathy. The patient develops myocarditis and expires on hospital day 5. Which of the following would have prevented this patient's presentation and decline?
Answer: Infectious diseases
Question: A 35-year-old woman with a history of Crohn disease presents for a follow-up appointment. She says that lately, she has started to notice difficulty walking. She says that some of her friends have joked that she appears to be walking as if she was drunk. Past medical history is significant for Crohn disease diagnosed 2 years ago, managed with natalizumab for the past year because her intestinal symptoms have become severe and unresponsive to other therapies. On physical examination, there is gait and limb ataxia present. Strength is 4/5 in the right upper limb. A T1/T2 MRI of the brain is ordered and is shown. Which of the following is the most likely diagnosis?
Answer: Neurology
Question: A 23-year-old G1 at 10 weeks gestation based on her last menstrual period is brought to the emergency department by her husband due to sudden vaginal bleeding. She says that she has mild lower abdominal cramps and is feeling dizzy and weak. Her blood pressure is 100/60 mm Hg, the pulse is 100/min, and the respiration rate is 15/min. She says that she has had light spotting over the last 3 days, but today the bleeding increased markedly and she also noticed the passage of clots. She says that she has changed three pads since the morning. She has also noticed that the nausea she was experiencing over the past few days has subsided. The physician examines her and notes that the cervical os is open and blood is pooling in the vagina. Products of conception can be visualized in the os. The patient is prepared for a suction curettage. Which of the following is the most likely cause for the pregnancy loss?
Answer: Obstetrics and gynecology
Question: An 8-month-old boy is brought to a medical office by his mother. The mother states that the boy has been very fussy and has not been feeding recently. The mother thinks the baby has been gaining weight despite not feeding well. The boy was delivered vaginally at 39 weeks gestation without complications. On physical examination, the boy is noted to be crying in his mother\u2019s arms. There is no evidence of cyanosis, and the cardiac examination is within normal limits. The crying intensifies when the abdomen is palpated. The abdomen is distended with tympany in the left lower quadrant. You suspect a condition caused by the failure of specialized cells to migrate. What is the most likely diagnosis?
Answer: Pediatrics\n
           """
# example + Question: & Answer:
# add in user content
# student
prompt_5 = "You are medical student taking a multiple choice exam. The knowledge of which of the following clinical specialties is the most helpful to answering the question: " + ", ".join(formatted_specialties)\
           + """\nHere are some examples:
Question: A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5\u00b0C (97.7\u00b0F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?
Answer: Cardiology
Question: A 68-year-old man comes to the physician because of recurrent episodes of nausea and abdominal discomfort for the past 4 months. The discomfort is located in the upper abdomen and sometimes occurs after eating, especially after a big meal. He has tried to go for a walk after dinner to help with digestion, but his complaints have only increased. For the past 3 weeks he has also had symptoms while climbing the stairs to his apartment. He has type 2 diabetes mellitus, hypertension, and stage 2 peripheral arterial disease. He has smoked one pack of cigarettes daily for the past 45 years. He drinks one to two beers daily and occasionally more on weekends. His current medications include metformin, enalapril, and aspirin. He is 168 cm (5 ft 6 in) tall and weighs 126 kg (278 lb); BMI is 45 kg/m2. His temperature is 36.4\u00b0C (97.5\u00b0F), pulse is 78/min, and blood pressure is 148/86 mm Hg. On physical examination, the abdomen is soft and nontender with no organomegaly. Foot pulses are absent bilaterally. An ECG shows no abnormalities. Which of the following is the most appropriate next step in diagnosis?
Answer: Gastroenterology
Question: A 6-year-old male who recently immigrated to the United States from Asia is admitted to the hospital with dyspnea. Physical exam reveals a gray pseudomembrane in the patient's oropharynx along with lymphadenopathy. The patient develops myocarditis and expires on hospital day 5. Which of the following would have prevented this patient's presentation and decline?
Answer: Infectious diseases
Question: A 35-year-old woman with a history of Crohn disease presents for a follow-up appointment. She says that lately, she has started to notice difficulty walking. She says that some of her friends have joked that she appears to be walking as if she was drunk. Past medical history is significant for Crohn disease diagnosed 2 years ago, managed with natalizumab for the past year because her intestinal symptoms have become severe and unresponsive to other therapies. On physical examination, there is gait and limb ataxia present. Strength is 4/5 in the right upper limb. A T1/T2 MRI of the brain is ordered and is shown. Which of the following is the most likely diagnosis?
Answer: Neurology
Question: A 23-year-old G1 at 10 weeks gestation based on her last menstrual period is brought to the emergency department by her husband due to sudden vaginal bleeding. She says that she has mild lower abdominal cramps and is feeling dizzy and weak. Her blood pressure is 100/60 mm Hg, the pulse is 100/min, and the respiration rate is 15/min. She says that she has had light spotting over the last 3 days, but today the bleeding increased markedly and she also noticed the passage of clots. She says that she has changed three pads since the morning. She has also noticed that the nausea she was experiencing over the past few days has subsided. The physician examines her and notes that the cervical os is open and blood is pooling in the vagina. Products of conception can be visualized in the os. The patient is prepared for a suction curettage. Which of the following is the most likely cause for the pregnancy loss?
Answer: Obstetrics and gynecology
Question: An 8-month-old boy is brought to a medical office by his mother. The mother states that the boy has been very fussy and has not been feeding recently. The mother thinks the baby has been gaining weight despite not feeding well. The boy was delivered vaginally at 39 weeks gestation without complications. On physical examination, the boy is noted to be crying in his mother\u2019s arms. There is no evidence of cyanosis, and the cardiac examination is within normal limits. The crying intensifies when the abdomen is palpated. The abdomen is distended with tympany in the left lower quadrant. You suspect a condition caused by the failure of specialized cells to migrate. What is the most likely diagnosis?
Answer: Pediatrics\n
           """

# inference GPT
api_key = "Your API Key"
client = OpenAI(api_key=api_key)

def call_gpt_api(messages, model):
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=256
    )
    return response.choices[0].message.content

category_1 = "output_path_cateogory_1"
category_2 = "output_path_cateogory_2"
category_3 = "output_path_cateogory_3"
category_4 = "output_path_cateogory_4"
category_5 = "output_path_cateogory_5"

# normal
with open(category_1, 'w') as output_file:
    messages = [{"role": "system", "content": prompt_1}]
    output_data = []
    for qa in sampled_data:
        if qa["topic_name"] is not None and qa["topic_name"].lower() in list_of_lower_specialties:
            line = {qa["question"]: qa["topic_name"]}
        else:
            question = qa["question"]
            chat_messages = messages.copy()
            chat_messages.append(
                {"role": "user", "content": question}
            )
            chat = call_gpt_api(chat_messages, "gpt-3.5-turbo")
            line = {qa["question"]: chat}
        output_data.append(line)
    json.dump(output_data, output_file, indent=4)

with open(category_1, 'r') as c1:
    result_1 = json.load(c1)
new_category_1 = "path_to_output_1"

with open(new_category_1, 'w') as nc_1:
    output_1 = []
    for cl_item in result_1:
        found = False
        for sp_item in list_of_specialties:
            for key, value in cl_item.items():
                if isinstance(value, str) and sp_item.lower() in value.lower():
                    cl_item[key] = sp_item
                    found = True
                    break
            if found:
                break
        if not found:
            cl_item[key] = "Others"
        output_1.append(cl_item)
    json.dump(output_1, nc_1, indent=4)

### SAME FOR OTHER PROMPTS
new_category_2 = "path_to_output_2"
new_category_3 = "path_to_output_3"
new_category_4 = "path_to_output_4"
new_category_5 = "path_to_output_5"

# Get voted results
def majority_vote(values):
    """
    Perform majority voting on a list of values.
    Requires at least 3 occurrences of a value to be chosen.
    If no value reaches majority criteria, `None` is returned.
    """
    count = Counter(values)
    for value, freq in count.items():
        if freq >= 3:  # Ensure value appears at least 3 times
            return value
    return "Others"  # No majority found

json_files = [new_category_1, new_category_2, new_category_3, new_category_4, new_category_5]
json_data = []

for file in json_files:
    with open(file, "r") as f:
        json_data.append(json.load(f))

num_dicts = len(json_data[0])
if not all(len(data) == num_dicts for data in json_data):
    raise ValueError("All JSON files must contain the same number of dictionaries.")

# Perform majority voting for each dictionary
voted_results = []
for i in range(num_dicts):
    dicts_at_index = [data[i] for data in json_data]
    combined_dict = {}
    keys = list(dicts_at_index[0].keys())
    for key_index, key in enumerate(keys):
        values = [list(d.values())[key_index] for d in dicts_at_index]
        combined_dict[key] = majority_vote(values)

    combined_dict[key] = majority_vote(values)
    voted_results.append(combined_dict)

# Save the voting results to a new JSON file
output_file = "voting_output_path"
with open(output_file, "w") as f:
    json.dump(voted_results, f, indent=4)

print(f"Majority voting results saved to {output_file}")
