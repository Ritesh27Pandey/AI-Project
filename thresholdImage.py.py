import cv2
img = cv2.imread("sampe2.jpeg")

grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresImg = cv2.threshold(grayImg,190,255,cv2.THRESH_BINARY) [1]
cv2.imwrite("thresholdImage2.jpg",thresImg)
