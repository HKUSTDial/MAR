from PIL import Image, ImageDraw, ImageFont


def merge(image_paths, save_path, text_type = 0):

    images = [Image.open(image_path) for image_path in image_paths]
    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images) + 55 * (len(image_paths)-1) + 25

    new_image = Image.new("RGB", (max_width, total_height), "white")

    y_offset = 0
    i = 0
    for image in images:
        draw = ImageDraw.Draw(new_image)
        if i==0:
            blank = 25
        else:
            blank = 50
        draw.rectangle([0, y_offset, max_width, y_offset + blank], fill="white")
        y_offset += blank
        text_y = y_offset - 20
        new_image.paste(image, (0, y_offset))

        y_offset += image.height
        draw.rectangle([0, y_offset, max_width, y_offset + 5], fill="black")
        y_offset += 5

        font_size = 12
        font = ImageFont.truetype('DejaVuSerif.ttf', font_size)
        if text_type == 0:
            text = 'Image '+ f'{i}'
        else:
            filename = image_paths[i].split('/')[-1]
            text = filename
        draw.text((2, text_y), text, fill='black', font=font)
        i += 1

    # save
    new_image.save(save_path)
