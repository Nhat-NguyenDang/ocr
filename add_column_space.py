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
				# text on topmost co-ordinate.
				# cv2.putText(img2, "Arrow tip", (x, y),
				# 				font, 0.5, (255, 0, 0))
				pass
			else:
				# text on remaining co-ordinates.
				# cv2.putText(img2, string, (x, y),
				# 		font, 0.5, (0, 255, 0))
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
	for index, coordinate in enumerate(x_coordinate[2:-15]):
		# for i in range (1, space+1):
		# 	img = np.insert(img, coordinate[0]+space*a, 240, axis=1)
		img = np.insert(img, [coordinate[0]+space*index], space_matrix, axis=1)
	return img


img2 = insert_space(img2, space=space)
print(img2.shape)
cv2.imshow('after', img2)
cv2.imwrite(working_dir+f'space_added_{space}.jpg', img2)
# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()




