# Image classification using Convolutional Neural Network

# Import the required libraries:
import cv2
import numpy as np
import math

# Capture video from the default camera (camera index 0):
cap = cv2.VideoCapture(0)

# Enter a loop to continuously process frames from the camera:
while(True):
    success,img=cap.read()

# Draw a green rectangle on the captured frame as a region of interest (ROI):    
    cv2.rectangle(img,(400,400),(50,50),(0,255,0),0)
    
# Crop the region within the rectangle:
    crop_img = img[50:400, 50:400]

# Convert the cropped image to grayscale:
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image:
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

# Apply thresholding to create a binary image
    success, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

# Find contours in the binary image:
    contours = cv2.findContours(thresh1.copy(), 
           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]

# Find the contour with the largest area:
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

# Draw a bounding rectangle around the largest contour within the cropped image:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

# Find the convex hull of the contour:
    hull = cv2.convexHull(cnt)

# Draw the original contour and its convex hull on an empty image:
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull],0,(0,0,255),0)

# Find the convexity defects in the contour:
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

# Iterate through the defects and calculate angles:
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # Calculate angles and distances
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        # Count the number of defects based on the angle
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        #dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img,start,end,[0,255,0],2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

# Display different text messages based on the number of defects (gestures) detected:
    if count_defects == 1:
        cv2.putText(img,"GESTURE ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        str = "GESTURE TWO"
        cv2.putText(img, str, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        #subprocess.Popen("sudo espeak GESTURE_TWO",shell=True)
    elif count_defects == 3:
        cv2.putText(img,"GESTURE THREE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"GESTURE FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"Hello World!!!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

# Display the processed frames with annotations:
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

# Exit the loop if the 'Esc' key is pressed:
    key = cv2.waitKey(1)
    if key == 27:
        break
# Release the webcam and close all windows when done:
cap.release()
cv2.destroyAllWindows()
