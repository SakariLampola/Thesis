# -*- coding: utf-8 -*-
"""
Detection of objects based on MobileNet and SSD

Created on Wed Nov 29 12:39:50 2017
@author: Sakari Lampola (based on code written by Adrian Rosebrock)
"""

import numpy as np
import cv2
import ImageObjectClasses as ioc

# load the serialized model from disk
NET = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", \
                               "MobileNetSSD_deploy.caffemodel")

def detect_objects(image, confidence_level):
    """
    Detect objects
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
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x_min, y_min, x_max, y_max) = box.astype("int")

            if idx == 1:
                objects.append(ioc.Aeroplane(x_min, y_min, x_max, y_max, confidence))
            elif idx == 2:
                objects.append(ioc.Bicycle(x_min, y_min, x_max, y_max, confidence))
            elif idx == 3:
                objects.append(ioc.Bird(x_min, y_min, x_max, y_max, confidence))
            elif idx == 4:
                objects.append(ioc.Boat(x_min, y_min, x_max, y_max, confidence))
            elif idx == 5:
                objects.append(ioc.Bottle(x_min, y_min, x_max, y_max, confidence))
            elif idx == 6:
                objects.append(ioc.Bus(x_min, y_min, x_max, y_max, confidence))
            elif idx == 7:
                objects.append(ioc.Car(x_min, y_min, x_max, y_max, confidence))
            elif idx == 8:
                objects.append(ioc.Cat(x_min, y_min, x_max, y_max, confidence))
            elif idx == 9:
                objects.append(ioc.Chair(x_min, y_min, x_max, y_max, confidence))
            elif idx == 10:
                objects.append(ioc.Cow(x_min, y_min, x_max, y_max, confidence))
            elif idx == 11:
                objects.append(ioc.DiningTable(x_min, y_min, x_max, y_max, confidence))
            elif idx == 12:
                objects.append(ioc.Dog(x_min, y_min, x_max, y_max, confidence))
            elif idx == 13:
                objects.append(ioc.Horse(x_min, y_min, x_max, y_max, confidence))
            elif idx == 14:
                objects.append(ioc.Motorbike(x_min, y_min, x_max, y_max, confidence))
            elif idx == 15:
                objects.append(ioc.Person(x_min, y_min, x_max, y_max, confidence))
            elif idx == 16:
                objects.append(ioc.PottedPlant(x_min, y_min, x_max, y_max, confidence))
            elif idx == 17:
                objects.append(ioc.Sheep(x_min, y_min, x_max, y_max, confidence))
            elif idx == 18:
                objects.append(ioc.Sofa(x_min, y_min, x_max, y_max, confidence))
            elif idx == 19:
                objects.append(ioc.Train(x_min, y_min, x_max, y_max, confidence))
            elif idx == 20:
                objects.append(ioc.TVMonitor(x_min, y_min, x_max, y_max, confidence))

    return objects
