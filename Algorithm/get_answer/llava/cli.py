import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import time
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

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles


    if args.dataset == 'original':
        qa_data = read_json('../NewsPersonQA/qa.json')
    else:
        qa_data = read_json('../NewsPersonQA/output/qa_with_prompt.json')

    ans_list = []
    # while True:
    for qa in qa_data:

        qa_id = qa['question_id']

        qa_img = qa['image']
        if args.dataset == 'original':
            qa_dl = qa['datalake']
            qa_text = qa['query'] + '.'
        else:
            qa_dl = ''
            qa_text = qa['text'] + '.'

        conv.clean_message()

        if args.dataset == 'original':
            image = load_image(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/{qa_img}.jpg')
        else:
            image = load_image(f'../NewsPersonQA/output/qa_retrieval/{qa_img}.jpg')

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)


        inp = f"{roles[0]}: {qa_text}"

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        ans_list.append({'question_id': qa_id, 'answer': outputs})
        print('question_id:', qa_id, 'answer:', outputs)
        # break

    # print('cost time:', time.time() - time1)
    if '7b' in args.model_path:
        if args.dataset == 'original':
            json_path = '../NewsPersonQA/output/qa_answer/llava-7b.json'
        else:
            json_path = '../NewsPersonQA/output/qa_answer/llava-7b_with_matching.json'
    else:
        if args.dataset == 'original':
            json_path = '../../../NewsPersonQA/output/qa_answer/llava-13b.json'
        else:
            json_path = '../../../NewsPersonQA/output/qa_answer/llava-13b_with_matching.json'


    save_to_json(ans_list, json_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-7b") # ./llava-v1.5-13b
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="../../3.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, default='with_prompt') #original, with_prompt
    args = parser.parse_args()
    main(args)
