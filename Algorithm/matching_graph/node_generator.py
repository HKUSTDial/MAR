import os
import json
from Algorithm.tool.utils import id_dict, read_json

for j in range(0, 110):
    folder_path = f"../NewsPersonQA/datalake/{j}/faces"
    json_file_path = f"../NewsPersonQA/datalake/{j}/nodes.json"
    raw_data = read_json(f'../NewsPersonQA/datalake/{j}/raw_data.json')
    id2names_dic = id_dict(raw_data, '_id', '_id', 'names')
    id2faces_num_dic = id_dict(raw_data, '_id', '_id', 'faces_num')


    def read_faces(folder_path):
        files = os.listdir(folder_path)
        files = sorted(files)
        cnt = 0
        nodes = []
        for file_name in files:
            if file_name.endswith(".jpg"):
                data_dict = {}
                data_dict['node_id'] = str('node_' + f"{cnt:06d}")
                data_dict['name'] = ''
                data_dict['is_credible'] = False
                data_dict['image'] = file_name.split('.')[0]
                data_dict['father_image'] = str(str(file_name.split('.')[0]).split('_')[0]) + '_' + str(str(file_name.split('.')[0]).split('_')[1])
                data_dict['neighbor_nodes'] = []
                nodes.append(data_dict)
                cnt += 1
        return nodes


    def generate_name(nodes):
        nodes_list = []
        for node in nodes:
            father_image_id = node["father_image"]
            name_list = []
            faces_num = 0
            try:
                name_list = id2names_dic[father_image_id]
                faces_num = id2faces_num_dic[father_image_id]
            except:
                pass
            if faces_num == 1 and len(name_list) == 1:
                node['is_credible'] = True
            node['name'] = name_list
            nodes_list.append(node)
        return nodes_list


    def save_to_json(data, json_file):
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


    nodes = read_faces(folder_path)
    nodes = generate_name(nodes)
    save_to_json(nodes, json_file_path)

    # node_linker.linker(json_file_path, json_file_path2)