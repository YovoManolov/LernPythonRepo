import cv2 as cv
import numpy as np
from skimage.color import label2rgb


# reading images
#img = cv.imread('Photos/group4.jpg')
img = cv.imread('Photos/group5.jpg')
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

# Connected components
op = cv.morphologyEx(255 - canny, cv.MORPH_GRADIENT, np.ones((5, 5)))
op = cv.erode(op, np.ones((2, 2)))
_, ccs = cv.connectedComponents(op, connectivity=4)

silhouette = label2rgb(ccs, colors=['white', 'gray'])
cv.imshow('Silhouette',  (255 * silhouette).astype(np.uint8))

cv.waitKey(0)
