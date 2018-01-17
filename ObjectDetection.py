# -*- coding: utf-8 -*-
"""
Detection of objects based on MobileNet and SSD

Created on Wed Nov 29 12:39:50 2017
@author: Sakari Lampola (based on code written by Adrian Rosebrock)
"""

import numpy as np
import cv2
import ImageClasses as ioc

# load the serialized model from disk
NET = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", \
                               "MobileNetSSD_deploy.caffemodel")

def get_hue_histogram(image, xmin, xmax, ymin, ymax):
    """
    get_hue_histogram(image, xmin, xmax, ymin, ymax)
        image: CV2 image
        xmin: bounding box horizontal minimum
        xmax: bounding box horizontal maximum
        ymin: bounding box vertical minimum
        ymax: bounding box vertical maximum

    Created on Wed Nov 29 09:08:16 2017
    @author: Sakari Lampola
    """
    image_roi = image[ymin:ymax, xmin:xmax, :]
    image_roi_hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_roi_hsv],[0],None,[8],[0,179])
    hist = hist / np.sum(hist)
    return hist

def detect_objects(image, time, confidence_level):
    """
    detect objects
        image: CV2 image
        confidence_level: range = 0...1.0, below the level objects are discarded

    Created on Wed Nov 29 09:08:16 2017
    @author: Sakari Lampola
    """
    # construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, \
                                 (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    NET.setInput(blob)
    detections = NET.forward()
    objects = []
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_level:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            class_type = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x_min, y_min, x_max, y_max) = box.astype("int")

            appearance = get_hue_histogram(image, x_min, x_max, y_min, y_max)
            objects.append(ioc.DetectedObject(time, class_type, x_min, x_max, \
                                              y_min, y_max, confidence, appearance))
            
    return objects
