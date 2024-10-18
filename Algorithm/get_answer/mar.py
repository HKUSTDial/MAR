import json
import string
import os
import base64
import openai
import time



def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def save_to_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


# 图片文件夹和JSON文件路径
images_folder = '../NewsPersonQA/output/qa_retrieval'
json_file_path = '../NewsPersonQA/qa.json'
answers_file_path = "../NewsPersonQA/output/qa_answer/mar.json"
graph_file_path = "../NewsPersonQA/output/qa_graph"

def get_query_type(query):
    query = query.lower()
    if query.startswith("who"):
        return 1
    elif query.startswith("is"):
        return 2
    elif query.startswith("how many"):
        return 3
    else:  #which
        return 4


# 函数：追加答案到JSON文件
def append_answer_to_file(question_id, answer, file_path):
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []

    data.append({"question_id": question_id, "answer": answer})

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def get_answer(question):
    question_text = question['query']
    query_type = get_query_type(question_text)
    qa_type = question['type']
    question_id = question['question_id']

    qa_id = question_id

    retrieval_graph = graph_file_path + '/'+ qa_id + '.json'
    node_datas = read_json(retrieval_graph)

    answer = ''
    name_dic = {}
    if query_type == 1:    # who
        for data in node_datas:  #Find the highest similarity
            if data['is_credible']:
                name = data['name'][0]
                if name in name_dic:
                    name_dic[name] += 1
                else:
                    name_dic[name] = 1

        if len(name_dic) < 1:
            return answer
        sorted_name_dict = dict(sorted(name_dic.items(), key=lambda item: item[1], reverse=True))
        name_list = list(sorted_name_dict.items())
        if len(name_list) == 1:
            answer = f'The person is most likely {name_list[0][0]}'
        elif len(name_list) > 1:
            answer = f'The person is most likely {name_list[0][0]}, also possibly {name_list[1][0]} '

    elif query_type == 2:  # is
        for data in node_datas:  #Find the highest similarity
            if data['is_credible']:
                name = data['name'][0].lower()
                if name in name_dic:
                    name_dic[name] += 1
                else:
                    name_dic[name] = 1

        if len(name_dic) < 1:
            return answer

        sorted_name_dict = dict(sorted(name_dic.items(), key=lambda item: item[1], reverse=True))
        name_list = list(sorted_name_dict.items())


        sentence = question_text.translate(str.maketrans("", "", string.punctuation)).lower()
        words = sentence.split()
        answer = 'No.'

        names = name_list[0][0].split()
        for name in names:
            if name in words:
                answer = 'Yes.'

        if len(name_dic) < 1:
            return answer

    elif query_type == 3 or query_type == 4:
        sentence = question_text.translate(str.maketrans("", "", string.punctuation)).lower()
        words = sentence.split()

        ans_list = []
        for data in node_datas:
            father_id = data['father_image']

            if data['is_credible']:
                name0 = data['name'][0].lower()
                names = name0.split()
                for name in names:
                    if name in words:
                        ans_list.append(father_id)
                        break

        if query_type == 4: #which
            i = 0
            for item in ans_list:
                if i == 0:
                    answer += item
                    i = 1
                else:
                    answer += ', '
                    answer += item
            answer += '.'
        else:  # how many
            answer = len(ans_list)


    return answer


def process_question(question):
    question_id = question['question_id']

    answer = get_answer(question)

    # append_answer_to_file(question_id, answer, answers_file_path)
    print(f"Processed and saved answer for question {question_id}.")
    return question_id, answer


if __name__ == "__main__":
    questions = read_json(json_file_path)
    output = []
    for question in questions:
        question_id, answer = process_question(question)
        output.append({"question_id": question_id, "answer": answer})
    save_to_json(output, answers_file_path)


    print("All questions have been processed.")
