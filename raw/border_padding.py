from PIL import Image
from PIL import ImageOps
import cv2

img = cv2.imread("temp/jp_test_img_3/last.jpg")
print(img.shape)
image = Image.open("temp/jp_test_img_3/last.jpg").convert("RGB")

image = ImageOps.pad(image,(1500,2050), color=(240,240,240))

image.save("padded_image.jpg")

img = cv2.imread("padded_image.jpg")
print(img.shape)

