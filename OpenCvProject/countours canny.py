import cv2 as cv
import numpy as np

# reading images
img = cv.imread('Photos/group 2.jpg')
cv.imshow('Group image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

countours, hierarchies = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print(f'{len(countours)} countours(s) found!')

cv.waitKey(0)
