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

def id_dict(datas, id_type, content1, content2):
    id_dict = {}
    for data in datas:
        if id_type == '_id':
            id = data[id_type]
            if content1 == '_id':
                for key, value in data[content2].items():
                    if isinstance(value, list):
                        if len(value) > 0:
                            id_dict[str(id + '_' + key)] = value
                    elif isinstance(value, int):
                        id_dict[str(id + '_' + key)] = value
                    elif isinstance(value, str):
                        id_dict[str(id + '_' + key)] = value
            else:
                for key, value in data[content1].items():
                    if isinstance(value, list):
                        if len(value) > 0:
                            id_dict[value] = str(id + '_' + key)
                    elif isinstance(value, int):
                        id_dict[value] = str(id + '_' + key)
        elif id_type == 'node_id':
                id_dict[data[content1]] = data[content2]
    return id_dict

def id2node(datas):
    id_dict = {}
    for data in datas:
        id_dict[data['node_id']] = data
    return id_dict
