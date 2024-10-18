from pathlib import Path
import numpy as np
import pandas as pd
import clip
import torch
import json
from PIL import Image as PILImage
from Algorithm.tool import images_merger
from Algorithm.tool.utils import id_dict, read_json, save_to_json
from Algorithm.tool.query_process import extract_name, get_query_type, extract_image
from Algorithm.matching_graph.node_linker import node_linker as node_linker
import re
import os

import subprocess

device = torch.device("cuda:3")
node_linker = node_linker()


def generate_prompt():
    target_path = '../NewsPersonQA/output/qa_with_prompt.json'
    qa_list = read_json('../NewsPersonQA/qa.json')
    qa_with_prompt_list = []


    for qa in qa_list:
        qa_result = []

        qa_id = qa['question_id']
        qa_dl = qa['datalake']
        qa_text = qa['query']
        qa_img = qa['image']
        qa_ans = qa['answer']
        qa_type = qa['type']

        print(qa_id)

        graph_num = int(qa_id)

        graph_num = f'{graph_num:06d}'
        graph_data = f'../NewsPersonQA/output/qa_graph/{graph_num}.json'

        retrieval_image_list = []
        retrieval_name_list = []


        node_data = read_json(graph_data)

        cnt = 0
        for node in node_data:
            if node['is_credible']:
                retrieval_image_list.append(node['image'])
                retrieval_name_list.append(node['name'][0])
                cnt += 1
        if cnt == 0:
            for node in node_data:
                if len(node['name']) > 0:
                    retrieval_image_list.append(node['image'])
                    retrieval_name_list.append(node['name'][0])

        query_with_prompt = 'Given that \"'
        if qa_type == 'single':
            for i in range(len(retrieval_image_list)):
                query_with_prompt = query_with_prompt + 'Image ' + str(i+1) + ' face ' + retrieval_image_list[i].split('_')[-1] + ' is ' + retrieval_name_list[i] + ', '
        else:
            for i in range(len(retrieval_image_list)):
                image_name = retrieval_image_list[i].split('_')[0] + '_' + retrieval_image_list[i].split('_')[1] +'.jpg'
                query_with_prompt = query_with_prompt + image_name + 'belongs to ' + retrieval_name_list[i] + ', '

        if qa_type == 'single':
            query_with_prompt = query_with_prompt + '\", Please answer based on the previous prompt and the faces feature: In the Image 0, ' + qa_text.lower()

        else:
            query_with_prompt = query_with_prompt + '\" Please answer based on the previous prompt and the faces feature: ' + qa_text.lower()


        qa_with_prompt_dict = {'question_id': qa_id, 'text': query_with_prompt, 'image': graph_num, 'graph': graph_num}
        qa_with_prompt_list.append(qa_with_prompt_dict)


    save_to_json(qa_with_prompt_list, target_path)


def generate_retrieval_data():

    node_max_len = 10
    qa_list = read_json('../NewsPersonQA/qa.json')

    temp_dl = 0
    for qa in qa_list:
        # time1 = time.time()
        qa_id = qa['question_id']
        qa_dl = qa['datalake']
        qa_text = qa['query']
        qa_img = qa['image']
        qa_ans = qa['answer']
        qa_type = qa['type']

        if qa_dl != temp_dl:
            node_linker.set_dl(qa_dl)
            temp_dl = qa_dl


        if qa_img == '':
            search_query = qa_text
        else:
            search_query = qa_text + ' [SEP] ' + qa_img

        print(qa_id)

        query_type = get_query_type(search_query)
        img_path = Path(f'../NewsPersonQA/datalake/{qa_dl}')
        img_features_path = Path(f'../NewsPersonQA/datalake/{qa_dl}/features/images')
        face_path = Path(f'../NewsPersonQA/datalake/{qa_dl}/faces')
        face_features_path = Path(f'../NewsPersonQA/datalake/{qa_dl}/features/faces')
        name_features_path = Path(f'../NewsPersonQA/datalake/{qa_dl}/features/names')

        # Load the features and the corresponding IDs
        name_ids = pd.read_csv(name_features_path / "name_ids.csv")
        name_ids = list(name_ids['name_id'])
        image_ids = pd.read_csv(img_features_path / "image_ids.csv")
        image_ids = list(image_ids['image_id'])
        face_ids = pd.read_csv(face_features_path / "image_ids.csv")
        face_ids = list(face_ids['image_id'])

        name_features = torch.from_numpy(np.load(name_features_path / "features.npy")).to(device)
        photo_features = torch.from_numpy(np.load(img_features_path / "features.npy")).to(device)
        face_features = torch.from_numpy(np.load(face_features_path / "features.npy")).to(device)

        raw_data = read_json(f'../NewsPersonQA/datalake/{qa_dl}/raw_data.json')
        # node_data = read_json(f'../NewsPersonQA/datalake/{qa_dl}/nodes.json')

        id2faces_num_dic = id_dict(raw_data, '_id', '_id', 'faces_num')



        def read_name_file(path):
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            data_dict = {}
            for value in data:
                for num, names in value['names'].items():
                    i = 0
                    if len(names) > 0:
                        for name in names:
                            data_dict[str(str(value['_id']) + '_' + str(num) + '_' + str(i))] = name
                            i += 1
            return data_dict


        def delete_mark_image(image_set, mark_image):
            image_set2 = []
            for image in image_set:
                img = image.split('_')[0] + '_' + image.split('_')[1]
                if img != mark_image:
                    image_set2.append(image)

            return image_set2


        model, preprocess = clip.load("ViT-B/32", device=device)

        name_in_query = extract_name(search_query)

        if len(name_in_query) == 0 and (query_type == 3 or query_type == 4) and qa_img == '':
            name_in_query = re.findall(r'named (.*?)\?', search_query)

        query_name = ''
        query_image = ''
        mark_image = ''  # The father_image corresponding to the images used to delete the query in the same data lake.

        if len(name_in_query) > 0:
            query_name = ' '.join(name_in_query)


        if '[SEP]' in search_query:
            query_image = extract_image(qa_dl, search_query)
            if query_type == 1 or query_type == 2:
                # remove itself
                query_image_file = query_image.split('/')[-1].split('.')[0]
                mark_image = query_image_file.split('_')[0] + '_' + query_image_file.split('_')[1]



        # Retrieval
        with torch.no_grad():

            if query_image != '':
                img = preprocess(PILImage.open(query_image)).unsqueeze(0).to(device)
                images_features = model.encode_image(img)

                img2img_sim = torch.cosine_similarity(images_features, face_features).squeeze(0)
                img2img_best = torch.sort(img2img_sim, descending=True)[0]
                img2img_best_idx = torch.argsort(img2img_sim, descending=True)


            elif query_name != '':
                name_encoded = model.encode_text(clip.tokenize(query_name).to(device))
                name_encoded /= name_encoded.norm(dim=-1, keepdim=True)
                qname_features = name_encoded
                txt2txt_sim = torch.cosine_similarity(qname_features, name_features).squeeze(0)
                txt2txt_best = torch.sort(txt2txt_sim, descending=True)[0]
                txt2txt_best_idx = torch.argsort(txt2txt_sim, descending=True)


                if query_type == 3 or query_type == 4:
                    desc_encoded = model.encode_text(clip.tokenize(search_query).to(device))
                    desc_encoded /= desc_encoded.norm(dim=-1, keepdim=True)
                    desc_features = desc_encoded
                    txt2img_sim = torch.cosine_similarity(desc_features, photo_features).squeeze(0)
                    txt2img_best = torch.sort(txt2img_sim, descending=True)[0]
                    txt2img_best_idx = torch.argsort(txt2img_sim, descending=True)

        face_results = set()


        flag1, flag2, flag3 = False, False, False

        i = 0
        topk = 3

        # Handle the retrieval results
        while not flag1 or not flag2 or not flag3:
            if query_image != '':
                flag1 = True
                flag2 = True
                sim3 = img2img_best[i]
                if sim3 < 0.7:
                    flag3 = True
                if not flag3:
                    idx = img2img_best_idx[i]
                    face_id = face_ids[idx]
                    face_results.add(face_id)

            elif query_name != '':
                flag3 = True
                sim1 = txt2txt_best[i]
                if sim1 < 0.5:
                    flag1 = True
                if not flag1:
                    idx = txt2txt_best_idx[i]
                    name_id = name_ids[idx]
                    name_id = '_'.join(name_id.split('_')[:2])
                    try:
                        face_num = id2faces_num_dic[name_id]
                    except:
                        face_num = 0

                    for j in range(face_num):
                        face_results.add(name_id + '_' + str(j))

                if query_type == 3 or query_type == 4:
                    sim2 = txt2img_best[i]
                    if sim2 < 0.3:
                        flag2 = True
                    if not flag2:
                        idx = txt2img_best_idx[i]
                        image_id = image_ids[idx]
                        try:
                            face_num = id2faces_num_dic[image_id]
                        except:
                            face_num = 0
                        for j in range(face_num):
                            face_results.add(image_id + '_' + str(j))
                else:
                    flag2 = True
            i += 1
            if query_image == '' and i > topk + 1:
                flag1 = flag2 = flag3 = True
            elif i > topk:
                flag1 = flag2 = flag3 = True

        face_results = delete_mark_image(face_results, mark_image)

        nodes = node_linker.linker(face_results, mark_image)


        if len(nodes) > node_max_len:
            nodes = nodes[:node_max_len]

        save_to_json(nodes, f'../NewsPersonQA/output/qa_graph/{qa_id}.json')

        # generate merge image
        merge_list = []
        merge_image_list = []

        if qa_type == 'single':
            merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/{qa_img}.jpg')

        # nodes = read_json(f'../NewsPersonQA/qa_graph/{qa_id}.json')
        for node in nodes:
            if node['is_credible']:
                merge_dict = {}
                merge_dict['name'] = node['name'][0]
                merge_dict['n_image'] = node['father_image']
                merge_dict['face'] = node['image'].split('_')[-1]
                merge_list.append(merge_dict)
                if qa_type == 'single':
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/' + node['father_image'] + '.jpg')
                else:
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images/' + node['father_image'] + '.jpg')

        if qa_type == 'single' and len(merge_image_list) == 1:
            for node in nodes:
                if len(node['name']) > 0:
                    merge_dict = {}
                    merge_dict['name'] = node['name'][0]
                    merge_dict['n_image'] = node['father_image']
                    merge_dict['face'] = node['image'].split('_')[-1]
                    merge_list.append(merge_dict)
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/' + node['father_image'] + '.jpg')
        elif qa_type == 'group' and len(merge_image_list) == 0:
            for node in nodes:
                if len(node['name']) > 0:
                    merge_dict = {}
                    merge_dict['name'] = node['name'][0]
                    merge_dict['n_image'] = node['father_image']
                    merge_dict['face'] = node['image'].split('_')[-1]
                    merge_list.append(merge_dict)
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images/' + node['father_image'] + '.jpg')
        if mark_image != '':
            images_merger.merge(merge_image_list, f'../NewsPersonQA/output/qa_retrieval/{qa_id}.jpg', 0)
        else:
            images_merger.merge(merge_image_list, f'../NewsPersonQA/output/qa_retrieval/{qa_id}.jpg', 1)


def graph2combine_image():
    node_max_len = 10
    qa_list = read_json('../NewsPersonQA/qa.json')

    for qa in qa_list:
        # time1 = time.time()
        qa_id = qa['question_id']
        qa_dl = qa['datalake']
        qa_text = qa['query']
        qa_img = qa['image']
        qa_ans = qa['answer']
        qa_type = qa['type']

        print(qa_id)

        nodes = read_json(f'../NewsPersonQA/output/qa_graph/{qa_id}.json')

        if len(nodes) > node_max_len:
            nodes = nodes[:node_max_len]

        # generate merge image
        merge_list = []
        merge_image_list = []

        if qa_type == 'single':
            merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/{qa_img}.jpg')

        # nodes = read_json(f'../NewsPersonQA/output/qa_graph/{qa_id}.json')
        for node in nodes:
            if node['is_credible']:
                merge_dict = {}
                merge_dict['name'] = node['name'][0]
                merge_dict['n_image'] = node['father_image']
                merge_dict['face'] = node['image'].split('_')[-1]
                merge_list.append(merge_dict)
                merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images/' + node['father_image'] + '.jpg')

        if qa_type == 'single' and len(merge_image_list) == 1:
            for node in nodes:
                if len(node['name']) > 0:
                    merge_dict = {}
                    merge_dict['name'] = node['name'][0]
                    merge_dict['n_image'] = node['father_image']
                    merge_dict['face'] = node['image'].split('_')[-1]
                    merge_list.append(merge_dict)
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images_wo_caption/' + node['father_image'] + '.jpg')
        elif qa_type == 'group' and len(merge_image_list) == 0:
            for node in nodes:
                if len(node['name']) > 0:
                    merge_dict = {}
                    merge_dict['name'] = node['name'][0]
                    merge_dict['n_image'] = node['father_image']
                    merge_dict['face'] = node['image'].split('_')[-1]
                    merge_list.append(merge_dict)
                    merge_image_list.append(f'../NewsPersonQA/datalake/{qa_dl}/images/' + node['father_image'] + '.jpg')

        if qa_type == 'single':
            images_merger.merge(merge_image_list, f'../NewsPersonQA/output/qa_retrieval/{qa_id}.jpg', 0)
        else:
            images_merger.merge(merge_image_list, f'../NewsPersonQA/output/qa_retrieval/{qa_id}.jpg', 1)


def fix_name():
    folder_path = '../NewsPersonQA/output/qa_graph'

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            flag = False
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                datas = read_json(file_path)
                for i in range(len(datas)):
                    if isinstance(datas[i]['name'], str):
                        datas[i]['name'] = [datas[i]['name']]
                        flag = True
            if flag:
                save_to_json(datas, file_path)
                print(file_path)


def get_answer(method):
    if method == 'mar':
        subprocess.run(['python', './get_answer/gpt.py'], check=True)
    else:
        subprocess.run(['python', './get_answer/gpt.py'], check=True)

# prompt generator
if __name__ == '__main__':
    generate_retrieval_data()
    generate_prompt()
    graph2combine_image()
    get_answer('gpt') #gpt, mar.   Note: llava needs to be run separately within its own project







