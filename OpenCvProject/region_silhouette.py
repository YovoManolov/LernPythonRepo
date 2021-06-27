import cv2 as cv

# reading images
img = cv.imread('Photos/group4.jpg')
cv.imshow('Group image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)

scale = 1
delta = 0
ddepth = cv.CV_16S

grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=1, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)
# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=1, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

sobelImage = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv.imshow('Sobel Demo - Simple Edge Detector', sobelImage)

threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', sobelImage)


cv.waitKey(0)
