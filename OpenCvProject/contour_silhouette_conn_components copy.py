import cv2 as cv
import numpy as np
from skimage.color import label2rgb


# reading images
#img = cv.imread('Photos/group4.jpg')
#img = cv.imread('Photos/group5.jpg')
img = cv.imread('Photos/human-thermal.jpg')
cv.imshow('Group image', img)

# Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

# Bluring
blur = cv.GaussianBlur(gray, (1, 1), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Canny Edges
canny = cv.Canny(blur, 30, 255)
cv.imshow('Canny Edges', canny)

op = cv.morphologyEx(255 - canny, cv.MORPH_GRADIENT, np.ones((5, 5)))
op = cv.erode(op, np.ones((2, 2)))
_, ccs = cv.connectedComponents(op, connectivity=4)

silhouette = label2rgb(ccs, colors=['white', 'gray'])
cv.imshow('Silhouette',  (255 * silhouette).astype(np.uint8))

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
# dilated = cv.dilate(canny, kernel)
# _, cnts, _ = cv.findContours(
#     dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# result = np.zeros_like(gray)
# cv.drawContours(result, [canny], 0, (255, 255, 255), cv.FILLED)

# cv.imshow('result', dilated)
cv.waitKey(0)
