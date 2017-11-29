# -*- coding: utf-8 -*-
"""
Analyze video frame by frame detecting objects and estimating states

Created on Wed Nov 29 13:44:02 2017
@author: Sakari Lampola
"""

import cv2
import ImageObjectDetection as iod

cap = cv2.VideoCapture("videos/CarsOnHighway001.mp4")
i_frame = 0;
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        print("Frame ",i_frame)
        i_frame = i_frame + 1;
        detected_objects = iod.detect_objects(frame, 0.2)
        for detected_object in detected_objects:
            # display the prediction
            label = "{}: {:.2f}%".format(detected_object.name, \
                     detected_object.confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (detected_object.x_min, detected_object.y_min), \
                          (detected_object.x_max, detected_object.y_max), 255, 2)
            y = detected_object.y_min - 15 if detected_object.y_min - 15 > 15 \
                else detected_object.y_min + 15
            cv2.putText(frame, label, (detected_object.x_min, y), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)        
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    