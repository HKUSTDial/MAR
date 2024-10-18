from PIL import Image, ImageDraw, ImageFont
import os
from utils import id_dict, read_json

image_path = "../../NewsPersonQA/raw_data/images_with_box"
output_folder = "../NewsPersonQA/raw_data/images_with_caption"

raw_data = read_json('../../NewsPersonQA/raw_data.json')

id2caption_dic = id_dict(raw_data, '_id', '_id', 'images')


def text_reshaper(words):
    # Initialize the counter and result text
    count = 0
    result_text = ""
    # Iterate through a list of words, adding a newline after every 30 words
    for word in words:
        result_text += word
        count += 1
        if count > 28 and word == ' ':
            result_text += "\n"
            count = 0
    return result_text

# Traverse images in the folder and perform face detection
for filename in os.listdir(image_path):
    if filename.endswith('.jpg'):
        image_path2 = os.path.join(image_path, filename)
        input_image = Image.open(image_path2)

        # Create a canvas with the same size as the original image
        output_image = Image.new('RGB', (input_image.size[0]+ 300,input_image.size[1]), color='white')

        # Paste the original image at the top of the new canvas.
        output_image.paste(input_image, (0, 0))

        # Create an object that can be used to draw text.
        draw = ImageDraw.Draw(output_image)

        text = ''
        try:
            text = id2caption_dic[filename.split('.')[0]]
        except:
            pass
        text = text_reshaper(text)
        font_size = 18
        font = ImageFont.truetype("times.ttf", font_size)

        # Calculate the size and position of the text so that it is centered at the bottom
        text_width, text_height = draw.textsize(text, font)
        image_width, image_height = output_image.size
        x = 230
        y = 5

        draw.text((x, y), text, fill='black', font=font)

        output_image.save(os.path.join(output_folder, filename))
