# Python code to find the co-ordinates of
# the contours detected in an image.
import numpy as np
import cv2

working_dir = "temp/test_imgs/"
space = 30
# Reading image
font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread(working_dir+'line_removed.jpg', cv2.IMREAD_COLOR)

# Reading same image in another
# variable and converting to gray scale.
img = cv2.imread(working_dir+'vertical_line.jpg', cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image
# (black and white only image).
_, threshold = cv2.threshold(img, 110, 240, cv2.THRESH_BINARY)

# Detecting contours in image.
contours, _= cv2.findContours(threshold, cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)

# Going through every contours found in the image.

x_coordinate = []
for cnt in contours :

	approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

	# draws boundary of contours.
	cv2.drawContours(img2, [approx], 0, (240, 240, 240), 7)

	# Used to flatted the array containing
	# the co-ordinates of the vertices.
	n = approx.ravel()
	i = 0

	for j in n :
		if(i % 2 == 0):
			x = n[i]
			y = n[i + 1]
			# print(f"{x}, {y}")

			# String containing the co-ordinates.
			string = str(x) + " " + str(y)
			# print(f"{x}, {y}")
			if(i == 0):
				pass
			else:
				x_coordinate.append([x,y])
		i = i + 1


# Showing the final image.
cv2.imshow('image2', img2)
def myFunc(x):
	return x[0]

x_coordinate.sort(key=myFunc)
print(img2.shape)

def insert_space(img, space):
	space_matrix = np.full((space, 1), 240)
	column = x_coordinate[2:-15]
	print(column)
	for index, coordinate in enumerate(column):
		# for i in range (1, space+1):
		# 	img = np.insert(img, coordinate[0]+space*a, 240, axis=1)
		img = np.insert(img, [coordinate[0]+space*index], space_matrix, axis=1)
	return img


img2 = insert_space(img2, space=space)
print(img2.shape)
cv2.imshow('after', img2)
# cv2.imwrite(working_dir+f'space_added_{space}.jpg', img2)

# Cut off white space and resize to 1024 x 768 (recommended size)
# img = cv2.imread('temp/test_imgs/space_added_30.jpg') # Read in the image and convert to grayscale
img3 = img2[20:-20,20:-20] # Perform pre-cropping
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
coords = cv2.findNonZero(gray) # Find all non-zero points (text)
x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
rect = img3[y-50:y+h+50, x-50:x+w+50] # Crop the image - note we do this on the original image

print(x, y, w, h)
#Resize to 1024x768 (recommended by GG Vision)
resized_image = cv2.resize(rect, (1024, 768), cv2.INTER_LANCZOS4)
# cv2.imwrite("resized_image.png", resized_image)
cv2.imshow("Resized", resized_image)

# Dilate
def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

dilated_image = thick_font(resized_image)
# cv2.imwrite(working_dir+"final.jpg", dilated_image)
# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()




