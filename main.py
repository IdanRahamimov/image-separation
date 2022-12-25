import cv2
import numpy as np

def main():
	# Read image
	img = cv2.imread('temp.png')
	# hh, ww = img.shape[:2]

	# colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
	# print(colors[count.argmax()])


	# threshold on white
	# Define lower and uppper limits
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.array([0, 0, 0])
	upper = np.array([255, 255, 40])

	# Create mask to only select black
	mask = cv2.inRange(hsv, lower, upper)
	print(mask)

	cv2.imshow('image window', mask)
	cv2.waitKey()

if __name__ == '__main__':
	main()