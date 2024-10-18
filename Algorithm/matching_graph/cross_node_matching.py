from pathlib import Path
import numpy as np
import pandas as pd
import json
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda:3")
model, preprocess = clip.load("ViT-B/32", device=device)
topk = 3

class node_matching():

    def __init__(self):
        self.dl = 0
        name_features_path = Path(f'../NewsPersonQA/datalake/{self.dl}/features/names')
        self.name_features = torch.from_numpy(np.load(name_features_path / "features.npy")).to(device)
        name_ids = pd.read_csv(name_features_path / "name_ids.csv")
        self.name_ids = list(name_ids['name_id'])
        face_features_path = Path(f'../NewsPersonQA/datalake/{self.dl}/features/faces')
        self.photo_features = torch.from_numpy(np.load(face_features_path / "features.npy")).to(device)
        face_ids = pd.read_csv(face_features_path / "image_ids.csv")
        self.face_ids = list(face_ids['image_id'])

    def set_dl(self, value):
        self.dl = value
        name_features_path = Path(f'../NewsPersonQA/datalake/{self.dl}/features/names')
        self.name_features = torch.from_numpy(np.load(name_features_path / "features.npy")).to(device)
        name_ids = pd.read_csv(name_features_path / "name_ids.csv")
        self.name_ids = list(name_ids['name_id'])
        face_features_path = Path(f'../NewsPersonQA/datalake/{self.dl}/features/faces')
        self.photo_features = torch.from_numpy(np.load(face_features_path / "features.npy")).to(device)
        face_ids = pd.read_csv(face_features_path / "image_ids.csv")
        self.face_ids = list(face_ids['image_id'])

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


    def matching(self, name_query, face_query, score, mark_image):

        with torch.no_grad():

            img = preprocess(Image.open(face_query)).unsqueeze(0).to(device)
            images_features = model.encode_image(img)

            if name_query != '':
                text_encoded = model.encode_text(clip.tokenize(name_query).to(device))
                text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

        similarities = torch.cosine_similarity(images_features,self.photo_features).squeeze(0)

        best_face = torch.sort(similarities, descending=True)[0]
        best_face_idx = torch.argsort(similarities, descending=True)


        if name_query != '':
            text_features = text_encoded
            similarities = torch.cosine_similarity(text_features, self.name_features).squeeze(0)
            best_name = torch.sort(similarities, descending=True)[0]
            best_name_idx = torch.argsort(similarities, descending=True)

        name_results = []
        if name_query != '':
            for i in range(len(best_name)):
                if best_name[i].item() > 0.9:
                    idx = best_name_idx[i]
                    name_id = self.name_ids[idx]
                    father_name = name_id.split('_')[0] + '_' + name_id.split('_')[1]
                    if mark_image == father_name:
                        continue
                    name_results.append(father_name)
                else:
                    break

        results = []
        for i in range(len(best_face)):
            test_score = round(best_face[i].item(), 4)
            if test_score > score and i < topk:
                idx = best_face_idx[i]
                face_id = self.face_ids[idx]
                father_name = face_id.split('_')[0] + '_' + face_id.split('_')[1]

                if i == 0:  # remove itself
                    i += 1
                    continue

                if father_name == mark_image:
                    continue

                if name_query != '' and father_name in name_results: #In cases of text input, it is necessary to consider the text.
                    results.append((face_id,test_score))
                    # print(result[0] * 0.1)
                elif name_query == '' or mark_image=='':            #No text input, or group QA.
                    results.append((face_id,test_score))
                i += 1
            else:
                break

        return results

