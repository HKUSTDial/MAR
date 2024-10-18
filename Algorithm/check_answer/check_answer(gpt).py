
import requests
import time
import re
import json

api_url = '' 
api_key = ''

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def save_to_json_line(data, json_file):
    with open(json_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def save_to_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

import re

def detect_space_number(sentence):
    pattern = r'\s+(\d+)'
    matches = re.findall(pattern, sentence)
    return matches


model = 'gpt4_with_matching' #llava
ans_path = '../NewsPersonQA/output/qa_answer/'+ model +'.json'
save_path = '../NewsPersonQA/output/qa_answer/final/' + model +'.json'


images_folder = '../NewsPersonQA/qa_group'
json_file_path = '../NewsPersonQA/output/qa_with_prompt.json'
answers_file_path = "../NewsPersonQA/output/qa_answer/gpt4_with_prompt/" 
error_ids = []



def get_answer(user_content, model="gpt-4"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {'model': model,
        'messages': [{'role': 'user','content': user_content}]
        }
    response = requests.post(api_url, headers=headers, json=data)
    response =response.json()
    return response['choices'][0]['message']['content']


if __name__ == '__main__':

    id2ans = {}

    standard_qa = read_json('../../NewsPersonQA/qa.json')
    output_qa = read_json(ans_path)

    single_data = {}
    results = []
    original_results = []

    for data in standard_qa:
        id2ans[data['question_id']] = {'question' :data['query'], 'answer': data['answer'], 'type': data['type']}

    for data in output_qa:

        qa_id = data['question_id']
        qa_question, standard_answer, qa_type = id2ans[qa_id]['question'], id2ans[qa_id]['answer'], id2ans[qa_id]['type']
        output_answer = data['answer']
        answer = 'false'
        value = 'false'


        if int(qa_id) > 18422:
            break

        num_list = detect_space_number(output_answer)
        is_credible = False
        if len(num_list) == 1:
            if str(num_list[0]) == '10':
                is_credible = True

        if qa_type == 'single':
            if output_answer.lower().find(standard_answer) != -1:
                answer = 'true'
                value = 'true'

            single_data = {'question_id': qa_id, 'model': model, 'question': qa_question,
                           'output_answer': output_answer, 'standard_answer': standard_answer, 'result': value,
                           'is_credible': is_credible}

            results.append(single_data)
            original_results = {'question_id': qa_id, 'answer': answer}
            print(single_data)
            continue


        text = f"According to the question, \"{qa_question}\", the standard answer is \"{standard_answer}\", while the judged answer is \"{output_answer}\". Please evaluate the judged answer for correctness and output it in the form of" + " {result: value}, where True indicates correct and False indicates incorrect."

        if qa_type == 'group':
            text += ' Please note that it is considered correct only when all the images are found exactly, without any more or fewer.'

        try:
            answer = str(get_answer(text))
        except:
            break

        matches = re.findall(r'\{([^}]*)\}', answer)

        result_dict = {}
        value = ''
        for match in matches:
            parts = match.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().lower()


        single_data = {'question_id': qa_id, 'model': model, 'question': qa_question, 'output_answer':output_answer, 'standard_answer': standard_answer, 'result': value, 'is_credible': is_credible}
        results.append(single_data)
        original_results = {'question_id': qa_id, 'answer': answer}
        print(single_data)

    save_to_json(results, save_path)





