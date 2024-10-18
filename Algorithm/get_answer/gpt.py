import json
import os
import base64
import openai
import time

# Set the API base URL
openai.api_base = ''
openai.api_base = ''

images_folder = '../../NewsPersonQA/qa_retrieval'
json_file_path = '../../NewsPersonQA/output/qa_with_prompt.json'
answers_file_path = "../../NewsPersonQA/output/qa_answer/gpt4_with_prompt.json"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_openai_api_with_retry(question, image_base64, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return call_openai_api(question, image_base64)
        except:
            print(f"API connection error, attempt {attempt+1} of {retries}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Failed to connect to OpenAI API after several attempts.")


# 函数：执行API调用
def call_openai_api(question, image_base64):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message['content'] if response.choices else "No response"

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

# 函数：处理问题
def process_question(question):
    image_id = question['image']
    image_path = os.path.join(images_folder, image_id + ".jpg")
    question_text = question['text']
    question_id = question['question_id']
    base64_image = encode_image(image_path)
    answer = call_openai_api_with_retry(question_text, base64_image)
    append_answer_to_file(question_id, answer, answers_file_path)
    print(f"Processed and saved answer for question {question_id}.")

# 主函数
def main():
    with open(json_file_path, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    for question in questions:
        process_question(question)
        break
    print("All questions have been processed.")

if __name__ == "__main__":
    main()
