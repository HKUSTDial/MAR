from Algorithm.tool.utils import id_dict, read_json, id2node
from cross_node_matching import node_matching
from collections import Counter
import os

LINKING_HIGHER_SCORE = 0.9
LINKING_HIGH_SCORE = 0.82
# LINKING_LOW_SCORE = 0.8


class node_linker():

    def __init__(self):
        self.dl = 0

        if not os.path.exists(f'../NewsPersonQA/datalake/{self.dl}/nodes.json'):
            os.system("python node_generator.py")
            print("Ran node_generator.py")
        else:
            print("nodes.json already exists.")

        raw_data = read_json(f'../NewsPersonQA/datalake/{self.dl}/raw_data.json')
        node_data = read_json(f'../NewsPersonQA/datalake/{self.dl}/nodes.json')
        self.id2names_dic = id_dict(raw_data, '_id', '_id', 'names')
        self.id2faces_num_dic = id_dict(raw_data, '_id', '_id', 'faces_num')
        self.face2nodeID_dict = id_dict(node_data, 'node_id', 'image', 'node_id')
        self.id2node_dic = id2node(node_data)
        self.nodeID2name_dic = id_dict(node_data, 'node_id', 'node_id', 'name')
        self.cross_node_matching = node_matching()

    def set_dl(self, value):
        self.dl = value
        self.cross_node_matching.set_dl(value)

        raw_data = read_json(f'../NewsPersonQA/datalake/{self.dl}/raw_data.json')
        node_data = read_json(f'../NewsPersonQA/datalake/{self.dl}/nodes.json')
        self.id2names_dic = id_dict(raw_data, '_id', '_id', 'names')
        self.id2faces_num_dic = id_dict(raw_data, '_id', '_id', 'faces_num')
        self.face2nodeID_dict = id_dict(node_data, 'node_id', 'image', 'node_id')
        self.id2node_dic = id2node(node_data)
        self.nodeID2name_dic = id_dict(node_data, 'node_id', 'node_id', 'name')

        if not os.path.exists(f'../NewsPersonQA/datalake/{self.dl}/nodes.json'):
            os.system("python node_generator.py")
            print("Ran node_generator.py")
        else:
            print("nodes.json already exists.")



    def link(self, nodes_data, mark_image):
        name = nodes_data["name"]
        face = nodes_data["image"]
        face_path = f'../NewsPersonQA/datalake/{self.dl}/faces/' + face +'.jpg'
        result = []
        if len(name) != 1:
            neighbor_face = self.cross_node_matching.matching("", face_path, LINKING_HIGHER_SCORE, mark_image)
        else:
            neighbor_face = self.cross_node_matching.matching(name, face_path, LINKING_HIGH_SCORE, mark_image)
        for item in neighbor_face:
            node_id = self.face2nodeID_dict[item[0]]
            score = item[1]
            result.append((node_id, score))
        return result


    def undire2dire(self, nodes_data):
        node_dict = {}
        i = 0
        for node_data in nodes_data:
            node_dict[node_data['node_id']] = i
            i += 1
        for node_data in nodes_data:
            neighbors = node_data['neighbor_nodes']
            if len(neighbors) > 0:
                for neighbor in neighbors: #Loop through each neighbor and add itself.
                    node_id = neighbor[0]
                    score = neighbor[1]   #The score of neighbor connection
                    try:
                        neighbor_neighbors = nodes_data[node_dict[node_id]]['neighbor_nodes']
                    except:
                        neighbor_neighbors = []
                    hasRecord = False

                    for j in range(len(neighbor_neighbors)):
                        if node_data['node_id'] == neighbor_neighbors[j][0]:
                            hasRecord = True
                            if neighbor_neighbors[j][1] < score:
                                neighbor_neighbors[j] = (neighbor_neighbors[j][0], score)
                    if not hasRecord:
                        neighbor_neighbors.append((node_data['node_id'], score))
                    try:
                        nodes_data[node_dict[node_id]]['neighbor_nodes'] = neighbor_neighbors
                    except:
                        pass
        return nodes_data



    def optimize(self, nodes_data):
        for n in range(len(nodes_data)):
            candidate_names = []
            name_score_dict = {}
            neighbor_names = set()
            if not nodes_data[n]['is_credible'] and len(nodes_data[n]['neighbor_nodes']) > 0: #The node with an uncertain name (equal to 0 or greater than 1), and with more than 0 neighboring nodes
                candidate_names.extend(nodes_data[n]['name'])
                for neighbor in nodes_data[n]['neighbor_nodes']:
                    nb_name = self.nodeID2name_dic[neighbor[0]]
                    if len(nb_name) > 0:
                        for name in nb_name:
                            candidate_names.append(name)
                            if name not in neighbor_names:
                                neighbor_names.add(name)
                                name_score_dict[name] = 0
                            name_score_dict[name] = max(name_score_dict[name], neighbor[1])

                counts = Counter(candidate_names)
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)


                try:
                    test = sorted_counts[0][1]
                except:
                    continue

                if sorted_counts[0][1] > 1:  # If the same name appears more than twice, it indicates that the person's name is "he/she."
                    nodes_data[n]['name'] = [sorted_counts[0][0]]
                    nodes_data[n]['is_credible'] = True
                else:
                    count = sorted_counts[0][1]
                    flag = True
                    i = 0
                    highest_name = sorted_counts[0][0]
                    highest_score = 0.0000
                    while flag:
                        if i >= len(sorted_counts):
                            break
                        try:
                            if sorted_counts[i][1] == count:
                                if name_score_dict[sorted_counts[i][0]] > highest_score:
                                    highest_name = sorted_counts[i][0]
                                    highest_score = name_score_dict[sorted_counts[i][0]]
                            else:
                                flag = False
                        except:
                            pass
                        i += 1
                    nodes_data[n]['name'] = [highest_name]
                    if highest_score > 0.9:
                        nodes_data[n]['is_credible'] = True
        return nodes_data


    def process(self, target_nodes, target_nodes_ids, mark_image):
        add_nodes_ids = set()
        add_nodes = []

        for i in range(len(target_nodes)):
            node_data1 = target_nodes[i]
            neighbors = self.link(node_data1, mark_image)

            target_nodes[i]["neighbor_nodes"] = neighbors

            if len(neighbors) > 0:
                for neighbor in neighbors:
                    if neighbor[0] not in target_nodes_ids:
                        add_nodes_ids.add(neighbor[0])

        for node_id in list(add_nodes_ids):
            add_nodes.append(self.id2node_dic[node_id])

        for i in range(len(add_nodes)):
            node_data2 = add_nodes[i]
            neighbors = self.link(node_data2, mark_image)
            add_nodes[i]["neighbor_nodes"] = neighbors

        output_nodes = target_nodes + add_nodes
        return output_nodes


    def linker(self, id_list, mark_image=''):
        nodes_id_list = []

        for id in id_list:
            try:
                nodes_id_list.append(self.face2nodeID_dict[id])
            except:
                pass

        # save_path = f'./output/nodes_data.json'
        target_nodes = []
        for node_id in nodes_id_list:
            target_nodes.append(self.id2node_dic[node_id])

        nodes = self.process(target_nodes, nodes_id_list, mark_image)
        nodes = self.undire2dire(nodes)
        nodes = self.optimize(nodes)

        # save_to_json(nodes, save_path)
        return nodes

