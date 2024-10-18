import spacy
import re

def extract_name(query):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    # extract the person name
    output = [ent.text for ent in doc.ents]
    return output


def replace_words(input_str, word_list, replace_word):
    for word in word_list:
        input_str = input_str.replace(word, replace_word)
    return input_str

w_list = ['What', 'Which', 'What\'s', 'Which', 'How many']

def get_query_type(query):
    query = query.lower()
    if query.startswith("who"):
        return 1
    elif query.startswith("is"):
        return 2
    elif query.startswith("how many"):
        return 3
    else:  #which
        return 4


def get_face(str):
    match = re.search(r'face(\d+)', str)
    if match:
        return match.group(1)

    return ''



def extract_image(datalake, input_str):

    start_index = input_str.find('[SEP]')
    image_name = input_str[start_index + len('[SEP]'):].strip()

    if 'face' in input_str:
        try:
            face_number = get_face(input_str)
            return f'../NewsPersonQA/datalake/{datalake}/faces/' + image_name + '_' + face_number+'.jpg'
        except:
            return f'../NewsPersonQA/datalake/{datalake}/images/' + image_name + '.jpg'
    else:
        return f'../NewsPersonQA/datalake/{datalake}/images/' + image_name + '.jpg'

