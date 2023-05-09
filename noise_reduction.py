import cv2

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def noise_reduce(img):
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
    no_noise = noise_removal(im_bw)
    return no_noise