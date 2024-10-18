from pathlib import Path
import json
import clip
import torch
import math
import numpy as np
import pandas as pd
import os

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def remove_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        # 删除文件
        os.remove(file_path)

import shutil
def delete_files_in_directory(directory):
    if not os.path.isdir(directory):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'There is some error in deleting file {file_path}. Massage: {e}')

def read_name_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_list = []

    for value in data:
        for num, names in value['names'].items():
            i = 0
            if len(names) > 0:
                for name in names:
                    data_list.append({
                        '_id': str(value['_id']) + '_' + str(num) + '_' + str(i),
                        'name': name
                    })
                    i += 1
    return data_list



def compute_clip_features(name_batch):
    # Load all the images from the files
    names = [names['name'] for names in name_batch]
    name_idx = clip.tokenize(names).to(device)
    # Preprocess all images
    # names_preprocessed = torch.stack([preprocess(name) for name in names]).to(device)

    with torch.no_grad():
        # Encode the images batch to compute the feature vectors and normalize them
        names_features = model.encode_text(name_idx)
        names_features /= names_features.norm(dim=-1, keepdim=True)
    # Transfer the feature vectors back to the CPU and convert to numpy
    return names_features.cpu().numpy()



for j in range(100, 110):
    batch_size = 16
    features_path = Path(f'../NewsPersonQA/datalake/{j}/features/names')
    name_files = read_name_file(f'../NewsPersonQA/datalake/{j}/raw_data.json')
    delete_files_in_directory(features_path)

    batches = math.ceil(len(name_files) / batch_size)

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i + 1}/{batches}")

        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the images for the current batch
                batch_files = name_files[i * batch_size: (i + 1) * batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the name IDs to a CSV file
                name_ids = [name_file['_id'] for name_file in batch_files]
                image_ids_data = pd.DataFrame(name_ids, columns=['name_id'])
                image_ids_data.to_csv(batch_ids_path, index=False)
            except:
                # Catch problems with the processing to make the process more robust
                print(f'Problem with batch {i}')



    # Load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)

    # Load all the image IDs
    image_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
    image_ids.to_csv(features_path / "name_ids.csv", index=False)