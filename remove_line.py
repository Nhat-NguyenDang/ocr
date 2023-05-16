import cv2
import numpy as np
import sys
from needed_functions import noise_reduce
import numpy as np

space = 30

def myFunc(x):
	return x[0]

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

#background color
background = 255
working_dir="temp/test_imgs/"


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nremove_line.py < path_to_image >')
        return -1
    
    image = cv2.imread(argv[0], cv2.IMREAD_COLOR)
    if image is None:
        print ('Error opening image: ' + argv[0])
        return -1
    cv2.imshow('raw', image)
    gray = noise_reduce(image)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, background, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow('gray', gray)
    #Dilate
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)) #(width, height)
    dilation = cv2.dilate(thresh, dilate_kernel, iterations = 1)

    cv2.imshow('dilation', dilation)

    # Remove vertical line
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 55)) #(width, height)
    vertical_lines = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray, [c], -1, (background,background,background), 12)
    cv2.imshow('vertical line', vertical_lines)
    _, threshold = cv2.threshold(vertical_lines, 110, 240, cv2.THRESH_BINARY)

    # Detecting contours in image.
    contours, _= cv2.findContours(threshold, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    # Going through every contours found in the image.

    x_coordinate = []
    for cnt in contours :
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        i = 0

        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                string = str(x) + " " + str(y)
                if(i == 0): pass
                else:
                    x_coordinate.append([x,y])
            i = i + 1

    x_coordinate.sort(key=myFunc)

    # Remove horizontal line
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, background, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)) #(width, height)
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray, [c], -1, (background,background,background), 12)

    cv2.imshow('horizontal line', detected_lines)
    # cv2.imshow('line delete', image)

    cv2.imshow('result', gray)
    # Save result
    # cv2.imwrite(working_dir+'line_removed.jpg', image)
    space_matrix = np.full((space, 1), background)
    for index, coordinate in enumerate(x_coordinate[2:-15]):
		# for i in range (1, space+1):
		# 	img = np.insert(img, coordinate[0]+space*a, 240, axis=1)
        gray = np.insert(gray, coordinate[0]+space*index, space_matrix, axis=1)

    cv2.imshow('after', gray)

    # Cut off white space and resize to 1024 x 768 (recommended size)
    # img = cv2.imread('temp/test_imgs/space_added_30.jpg') # Read in the image and convert to grayscale
    pre_cropping = gray[20:-20,20:-20] # Perform pre-cropping
    pre_cropping = 255*(pre_cropping < 128).astype(np.uint8) # To invert the text to white
    pre_cropping = cv2.morphologyEx(pre_cropping, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(pre_cropping) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = gray[y-50:y+h+50, x-50:x+w+50] # Crop the image - note we do this on the original image

    #Resize to 1024x768 (recommended by GG Vision)
    resized_image = cv2.resize(rect, (1024, 768), cv2.INTER_LANCZOS4)
    # cv2.imwrite("resized_image.png", resized_image)
    dilated_image = thick_font(resized_image)
    cv2.imshow("Last", dilated_image)
    cv2.imwrite(working_dir+"final.png", dilated_image)
    
    cv2.waitKey()
    cv2.destroyWindow('hahaha')


if __name__ == "__main__":
    main(sys.argv[1:])

