import joblib
import json
import numpy as np
import base64
import cv2
import sklearn
import pywt
from wavelet import w2d
import io
from imageio import imread
from PIL import Image

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path= None):
    valid_image = cropped_image_if_2_eyes(file_path, image_base64_data)
    if valid_image == True:
        imgs = moustache(file_path, image_base64_data)


    result = []

    for img in imgs:
        scaled_img = cv2.resize(img, (50, 50))
        blur = cv2.GaussianBlur(img, (5, 5), 0) # check if changing to img and then resize makes a diff or not
        img_har = w2d(blur, 'db5', 6)
        scaled_img_har = cv2.resize(img_har, (50,50))
        combined_img = np.vstack((scaled_img.reshape(50 * 50 * 3, 1), scaled_img_har.reshape(50 * 50, 1))) #converts to 10,000 rows and 1 column
        len_image_array = (50 * 50 * 3) + (50 *50)
        final = combined_img.reshape(1, len_image_array).astype(float) #converts to 10,000 columns and 1 row



        result.append({
            'class' : class_number_to_name(__model.predict(final)[0]),
            # 'class_prob' : np.round(__model.predict_proba(final)* 100, 2).tolist()[0],
            # 'class_dictionary': __class_name_to_number


        }) # This might just be returning one image anyways even if two faces, check this


    return result


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_number_to_name
    global __class_name_to_number

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()} #basically take dic items and reverse key and value

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl','rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# def get_cv2_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

def stringToImage(b64string):
    imgdata = base64.b64decode(b64string)
    image = Image.open(io.BytesIO(imgdata))
    return image

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def get_cv2_image_from_base64_string(b64str):
    img_PIL = stringToImage(b64str)
    img = toRGB(img_PIL)
    return img

def cropped_image_if_2_eyes(image_path, image_base64_data):

    eye_check = True
    #we are using this block to check if image has two eyes or not

    face_cascade = cv2.CascadeClassifier('./opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier("./opencv/sources/data/haarcascades/haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eye_check = True
        else:
            eye_check = False

    return eye_check

def moustache(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier("./opencv/sources/data/haarcascades/haarcascade_eye.xml")

    img = cv2.imread(image_path)

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
            return None
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cropped_faces = []

        for (x,y,w,h) in faces:
            y1 = int(y + (h/2))
            h1 = int(h/2)
            x1 = int(x + w/4)
            w1 = int((3*w)/4)
            roi_gray_moustache = gray[y1:y1+h1, x1:x1+w1].copy()
            roi_color_moustache = img[y1:y1+h1, x1:x1+w1].copy()
            cropped_faces.append(roi_color_moustache)
        return cropped_faces


def get_b64_image_chevron():
    with open("b64.txt") as f:
        return f.read() #here the image is returned as a string


#the code below is only when util is running directly
if __name__ == "__main__":
    load_saved_artifacts()
    #print(classify_image(get_b64_image_chevron(), None))
    #print(classify_image(None, "./test_images/handle_bar_1.jpg"))
    # print(classify_image(None, "./test_images/handle_bar_2.jpg"))
    #print(classify_image(None, "./test_images/horseshoe_1.jpg"))
    # print(classify_image(None, "./test_images/horseshoe_2.jpg"))

