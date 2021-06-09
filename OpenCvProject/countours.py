import cv2 as cv
import numpy as np

# reading images
img = cv.imread('Photos/group 2.jpg')
cv.imshow('Cats', img)

# blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('Blank', blank)

# Countours are not edges
# Countours are usefull in object detection and recognition

#cv.imshow('Group 1', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

countours, hierarchies = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print(f'{len(countours)} countours(s) found!')

# cv.drawContours(blank, countours, -1, (0, 0, 255), 1)
# cv.imshow('Countours Drawn', blank)

cv.waitKey(0)
