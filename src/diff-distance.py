# Two images capture a moving object
# find the distance the object travelled
# adapted from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
# and https://stackoverflow.com/questions/56781635/find-extreme-outer-points-in-image-with-python-opencv

import cv2
import numpy as np

im0 = cv2.imread('images/img1.jpg')
im1 = cv2.imread('images/img2.jpg')

# subtract images, grayscale, Gaussian blur, threshold
diff = cv2.absdiff(im0, im1)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]

# find bounds
x, y, w, h = cv2.boundingRect(thresh)
# Obtain outer coordinates
left = (x, np.argmax(thresh[:, x]))             #
right = (x+w-1, np.argmax(thresh[:, x+w-1]))    #

# Draw edge dots onto image
cv2.circle(diff, left, 8, (0, 50, 255), -1)
cv2.circle(diff, right, 8, (0, 255, 255), -1)

# add diff text to image
cv2.putText(diff, 'distance: {}px'.format(right[0] - left[0]), (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
# save the image
cv2.imwrite("images/result-diff-distance.jpg", diff)
