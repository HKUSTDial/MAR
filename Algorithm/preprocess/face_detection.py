from deepface import DeepFace
from PIL import Image
import os
import cv2

error_list = []
remove_list = []

def resize_image(image, target_short_edge=100):
    original_image = image
    width, height = original_image.size
    # Calculate the size after proportional scaling
    if width < height:
        new_width = target_short_edge
        new_height = int(target_short_edge * (height / width))
    else:
        new_width = int(target_short_edge * (width / height))
        new_height = target_short_edge

    # Using the Pillow library for image resizing
    resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)

    return resized_image


def detect_and_save_faces(image_path, image_box_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    img = Image.open(image_path)
    img2 = cv2.imread(image_path)
    img_name = image_path.split('/')[-1].split('\\')[-1].split('.')[0]
    faces = []
    # Using the DeepFace
    try:
        faces = DeepFace.analyze(img_path=image_path, actions=['face_detection'], detector_backend='ssd', enforce_detection=False)
    except:
        try:
            faces = DeepFace.analyze(img_path=image_path, actions=['face_detection'], detector_backend='mtcnn',
                                     enforce_detection=False)
        except:
            error_list.append(image_path)


    for i, face in enumerate(faces):
        # Retrieve the coordinates of each face
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

        if x == 0 and y == 0 and w == 224 and h == 224:  #There are no faces in the image.
            remove_list.append(image_path)
            return

        # Extract faces from the original image
        face_img = img.crop((x, y, x + w, y + h))

        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if y+h > 170:
            cv2.putText(img2, str(i), (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(img2, str(i), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        face_img = resize_image(face_img)

        # Save face image
        save_path = f"{output_folder}/{img_name}_{i}.jpg"
        face_img.save(save_path)
    box_path = image_box_path + '/' + img_name+'.jpg'
    cv2.imwrite(box_path, img2)
        # print(f"Face {i + 1} saved at: {save_path}")


image_path = "../../NewsPersonQA/raw_data/images"
image_box_path = "../../NewsPersonQA/raw_data/images_with_box"
output_folder = "../NewsPersonQA/faces"


for filename in os.listdir(image_path):
    if filename.endswith('.jpg'):
        image_path2 = os.path.join(image_path, filename)
        detect_and_save_faces(image_path2, image_box_path, output_folder)