import cv2
import numpy as np
import sys

#background color
background = 240
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
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, background, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow('raw', image)
    
    #Dilate
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)) #(width, height)
    dilation = cv2.dilate(thresh, dilate_kernel, iterations = 1)

    cv2.imshow('dilation', dilation)

    # Remove vertical line
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 55)) #(width, height)
    detected_lines = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (background,background,background), 7)

    cv2.imshow('vertical line', detected_lines)
    cv2.imwrite(working_dir+'vertical_line.jpg', detected_lines)
    # cv2.imshow('after vertical', image)


    # Remove horizontal line
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, background, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)) #(width, height)
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (background,background,background), 7)

    cv2.imshow('horizontal line', detected_lines)

    cv2.imshow('line delete', image)

    #Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,6))
    result = 255 - cv2.morphologyEx(255 - image[:, :1120, :], cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    result = np.column_stack((result, image[:, 1120:, :]))


    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyWindow('hahaha')

    # Save result
    cv2.imwrite(working_dir+'line_removed.jpg', result)


if __name__ == "__main__":
    main(sys.argv[1:])

