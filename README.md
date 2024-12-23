# MAR: Matching-Augmented Reasoning for Enhancing Visual-based Entity Question Answering

[📃[Paper Link]](https://aclanthology.org/2024.emnlp-main.91.pdf)


![MAR Overview](./assets/model.jpg)


## 🔥 News
[24/9/20] 🎊 Our paper has been accepted by EMNLP 2024 (main).

## 📖 Abstract

A multimodal large language model (MLLM) may struggle with answering visual-based (personal) entity questions (VEQA), such as “who is A?” or “who is A that B is talking to?” for various reasons, e.g., the absence of the name of A in the caption or the inability of MLLMs to recognize A, particularly for less common entities. Furthermore, even if the MLLM can identify A, it may refrain from answering due to privacy concerns. In this paper, we introduce a novel method called Matching-Augmented Reasoning (MAR) to enhance VEQA. Given a collection of visual objects with captions, MAR preprocesses each object individually, identifying faces, names, and their alignments within the object. It encodes the information and stores their vector representations in the database. When handling VEQA, MAR retrieves matching faces and names and organizes these entities into a matching graph. MAR then derives the answer to the query by reasoning over this matching graph. Extensive experiments show that MAR significantly improves VEQA compared with the state-of-the-art methods using MLLMs.


## 📜 Contribution

- We study VEQA, an important and commonly used subset of VQA, but it is not fully explored.
- We propose matching graphs that can capture the relationships of the same entities over multiple captioned visual objects. Based on a matching graph, we proposed matching augmenting reasoning (MAR), to effectively answer a VEQA.
- Given the lack of VEQA dataset focusing on the personal entity, we construct a new benchmark NewsPersonQA including 235k images and 6k QA pairs.
- We conduct extensive experiments to show that MAR > MLLMs + RAG > MLLMs, where RAG is to feed the retrieved matching graph to MLLMs. 

## 📁 File Structure

- `Algorithm`: Contains the core code for the proposed algorithm.
  - `check_answer`: Includes methods to validate the answers generated by the model, using both the GPT agent and string matching methods.
  - `get_answer`: Contains the method to generate answers from the model.
  - `matching_graph`: The core code of our proposal, focusing on the matching graph.
  - `preprocess`: Includes methods to construct the `NewsPersonQA` benchmark from the [GoodNews](https://github.com/furkanbiten/GoodNews) dataset.
  - `tool`: Provides utility functions for file handling, image processing, and other tasks.
  - `main.py`: The pipeline of the **MAR**.
  - `qa.json`: Contains the QA pairs.

- `NewsPersonQA`: Our proposed benchmark dataset. The original data, feature encodings, graph results, and retrieval results can be accessed from [NewsPersonQA](https://pan.baidu.com/s/1s661H9gUEYsqI7PiNxs0PQ?pwd=u759) (**Password:** u759).
  - `datalake`: The main folder containing the dataset, which includes 110 datalakes.
    - `face`: Pre-extracted faces from the images.
    - `feature`: Feature vectors for names, faces, and images, generated using CLIP encoding. You can also generate these using `../Algorithm/preprocess/everything2feature.py`.
    - `images`: Contains images with captions from the datalake.
    - `images_wo_caption`: Contains images without captions from the datalake.
    - `nodes.json`: Contains the initial nodes for each datalake. You can also generate this file using `../Algorithm/main.py`.
    - `raw_data.json`: Contains the original news articles for each datalake.
  - `output`: The folder for output results from the model.
  - `raw_data`: Contains raw data from [GoodNews](https://github.com/furkanbiten/GoodNews), used for dataset construction.
  - `qa.json`: Contains all QA pairs for the dataset.

### Attention
1. Due to the large size and multimodal nature of the NewsPersonQA dataset, please download it from the cloud drive: [link](https://pan.baidu.com/s/1s661H9gUEYsqI7PiNxs0PQ?pwd=u759) (**Password:** u759), and replace the files in the project. In addition to the original dataset, the zip package downloaded from the cloud also includes the results retrieved using a matching graph and the generated prompts for input into MLLM (qa_with_prompt.json).
2. To help readers clearly and intuitively understand the process of retrieval and graph node linking through similarity matching, our open-source code directly uses torch.cosine_similarity to replace the operations described in our paper that were performed on the Faiss database (including storage and retrieval based on a flat index). This approach aligns with the principle of using Faiss flat index retrieval as described in the paper.

## 🛠️Environment Setup
To set up the environment for MAR, ensure you have Anaconda installed and then follow these steps:

1. **Create and activate a new environment:**
```
conda create -n mar python=3.10.13
conda activate mar
```

2. **Install the necessary packages:**
```
pip install -r requirements.txt
```
## ✏️Citation

If you find our work useful or inspiring, please kindly cite:

```
@inproceedings{
zhang2024mar,
title={{MAR}: Matching-Augmented Reasoning for Enhancing Visual-based Entity Question Answering},
author={Zhang, Zhengxuan and Wu, Yin and Luo, Yuyu and Tang, Nan},
booktitle ={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
url={https://aclanthology.org/2024.emnlp-main.91},
pages={1520--1530}
}
```