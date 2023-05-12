import cv2
import base64
import numpy as np


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_reduce(img):
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(im_bw, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def encode_to_base64(img):
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string
