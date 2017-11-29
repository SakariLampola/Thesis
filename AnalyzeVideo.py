# -*- coding: utf-8 -*-
"""
Analyze video frame by frame detecting objects and estimating states

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import cv2
import ImageObjectDetection as iod

cap = cv2.VideoCapture("videos/CarsOnHighway001.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    detected_objects = iod.detect_objects(frame, 0.2)
    for detected_object in detected_objects:
        detected_object.print_attributes()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

image = cv2.imread("images/example_01.jpg")
detected_objects = iod.detect_objects(image, 0.2)
for detected_object in detected_objects:
    detected_object.print_attributes()
    