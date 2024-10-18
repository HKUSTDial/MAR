from pathlib import Path
import clip
import torch
from PIL import Image
import math
import numpy as np
import pandas as pd
import os

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def remove_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
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


# Function that computes the feature vectors for a batch of images
def compute_clip_features(images_batch):
    images = [Image.open(image_file) for image_file in images_batch]

    # Preprocess all images
    images_preprocessed = torch.stack([preprocess(image) for image in images]).to(device)

    with torch.no_grad():
        # Encode the images batch to compute the feature vectors and normalize them
        images_features = model.encode_image(images_preprocessed)
        images_features /= images_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return images_features.cpu().numpy()

batch_size = 16
for j in range(100, 110):
    features_path = Path(f'../NewsPersonQA/datalake/{j}/features/faces')
    images_path = Path(f'../NewsPersonQA/datalake/{j}/faces')

    delete_files_in_directory(features_path)

    # List all JPGs in the folder
    images_files = list(images_path.glob("*.jpg"))

    # Compute how many batches are needed
    batches = math.ceil(len(images_files) / batch_size)

    # Print some statistics
    print(f"images found: {len(images_files)}")

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i + 1}/{batches}")

        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the images for the current batch
                batch_files = images_files[i * batch_size: (i + 1) * batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the image IDs to a CSV file
                image_ids = [image_file.name.split(".")[0] for image_file in batch_files]
                image_ids_data = pd.DataFrame(image_ids, columns=['image_id'])
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
    image_ids.to_csv(features_path / "image_ids.csv", index=False)