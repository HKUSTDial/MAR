
import requests
import time
import openai
import re
import json


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


model = 'gpt4_with_matching'
save_path = '../NewsPersonQA/output/qa_answer/final/' + model +'.json'


images_folder = '../NewsPersonQA/qa_retrieval'
json_file_path = '../NewsPersonQA/output/qa_with_prompt.json'
answers_file_path = "../NewsPersonQA/output/qa_answer/gpt4_with_prompt/"
error_ids = []


if __name__ == '__main__':

    id2ans = {}

    standard_qa = read_json('../NewsPersonQA/qa.json')

    single_data = {}
    results = []
    original_results = []

    for data in standard_qa:
        qa_id = data['question_id']

        if int(qa_id) > 18434:
            break

        answer = ''
        value = 'false'

        qa_question, standard_answer, qa_type = data['query'], data['answer'], data['type']

        try:
            ans = read_json(answers_file_path + '/' + qa_id + '.json')
            output_answer = ans['answer']

        except:
            single_data = {'question_id': qa_id, 'model': model, 'question_type': qa_type, 'question': qa_question,
                           'output_answer': output_answer, 'standard_answer': standard_answer, 'result': value}
            results.append(single_data)
            continue



        standard_answer2 = standard_answer.lower().split()
        for name in standard_answer2:
            if output_answer.lower().find(name) != -1:
                value = 'true'

        single_data = {'question_id': qa_id, 'model': model, 'question': qa_question, 'output_answer':output_answer, 'standard_answer': standard_answer, 'result': value}
        results.append(single_data)
        print(single_data)

    save_to_json(results, save_path)





