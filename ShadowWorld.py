# -*- coding: utf-8 -*-
"""
ShadowWorld version 2.0
Created on Mon Jan 29 08:30:49 2018

@author: Sakari Lampola
"""
# Imports----------------------------------------------------------------------
import numpy as np
import cv2
import time
import random as rnd
import pyttsx3
from math import atan, cos, sqrt, inf, tan
from scipy.optimize import linear_sum_assignment
# Hyperparameters--------------------------------------------------------------
BORDER_WIDTH = 30 # Part of video window for special pattern behaviour
CONFIDENFE_LEVEL_CREATE = 0.80 # How confident must be to create new pattern
CONFIDENFE_LEVEL_UPDATE = 0.20 # How confident must be to update pattern
PATTERN_ALFA = 200.0 # Pattern initial location error variance
PATTERN_BETA = 10000.0 # Pattern initial velocity error variance
PATTERN_Q = np.array([200.0]) # Pattern measurement variance
PATTERN_R = np.array([[0.1, 0.0],
                      [0.0, 1.0]]) # Pattern state equation covariance
PATTERN_C = np.array([[1.0, 0.0]]) # Pattern measurement matrix
RETENTION_COUNT_MAX = 30 # How many frames pattern is kept untedected
SIMILARITY_DISTANCE = 0.4 # Max distance for detection-pattern similarity
# Other constants--------------------------------------------------------------
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
               "dog", "horse", "motorbike", "person", "potted plant", "sheep",
               "sofa", "train", "tv monitor"]
NET = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", \
                               "MobileNetSSD_deploy.caffemodel")
# Classes----------------------------------------------------------------------
class Body:
    """
    A physical entity in the world
    """
    def __init__(self, world, pattern):
        """
        Initialization
        """
        # References
        self.world = world
        self.pattern = pattern
        self.events = []
        self.traces = []
        # Attributes
        self.set_class_attributes(pattern.class_id)
        self.x, self.y, self.z = self.coordinates_from_pattern()
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

    def set_class_attributes(self, class_id):
        """
        Sets class specific attributes
        """
        self.class_id = class_id
        if class_id == 1: # Aeroplane
            self.height_min, self.height_mean, self.height_max = 1.5, 2.0, 6.0
            self.width_min, self.width_mean, self.width_max = 2.0, 20.0, 80.0
            self.length_min, self.length_mean, self.length_max = 3.0, 20.0, 80.0
            self.velocity_max, self.acceleration_max = 300.0, 10.0
        if class_id == 2: # Bicycle
            self.height_min, self.height_mean, self.height_max = 0.6, 1.1, 1.4
            self.width_min, self.width_mean, self.width_max = 0.4, 0.6, 0.8
            self.length_min, self.length_mean, self.length_max = 1.2, 1.5, 1.7
            self.velocity_max, self.acceleration_max = 13.0, 1.0
        if class_id == 3: # Bird
            self.height_min, self.height_mean, self.height_max = 0.1, 0.25, 2.8
            self.width_min, self.width_mean, self.width_max = 0.05, 0.5, 1.2
            self.length_min, self.length_mean, self.length_max = 0.1, 0.2, 1.2
            self.velocity_max, self.acceleration_max = 80.0, 100.0
        if class_id == 4: # Boat
            self.height_min, self.height_mean, self.height_max = 1.0, 2.0, 20.0
            self.width_min, self.width_mean, self.width_max = 0.6, 3.0, 60.0
            self.length_min, self.length_mean, self.length_max = 1.5, 3.0, 350.0
            self.velocity_max, self.acceleration_max = 20.0, 5.0
        if class_id == 5: # Bottle
            self.height_min, self.height_mean, self.height_max = 0.1, 0.25, 0.5
            self.width_min, self.width_mean, self.width_max = 0.05, 0.1, 0.3
            self.length_min, self.length_mean, self.length_max = 0.05, 0.1, 0.3
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 6: # Bus
            self.height_min, self.height_mean, self.height_max = 2.5, 2.8, 5.0
            self.width_min, self.width_mean, self.width_max = 2.3, 2.5, 2.8
            self.length_min, self.length_mean, self.length_max = 6.0, 15.0, 30.0
            self.velocity_max, self.acceleration_max = 30.0, 2.0
        if class_id == 7: # Car
            self.height_min, self.height_mean, self.height_max = 1.3, 1.5, 2.0
            self.width_min, self.width_mean, self.width_max = 1.6, 1.8, 2.0
            self.length_min, self.length_mean, self.length_max = 3.5, 4.3, 5.5
            self.velocity_max, self.acceleration_max = 50.0, 10.0
        if class_id == 8: # Cat
            self.height_min, self.height_mean, self.height_max = 0.2, 0.3, 0.4
            self.width_min, self.width_mean, self.width_max = 0.1, 0.15, 0.2
            self.length_min, self.length_mean, self.length_max = 0.3, 0.4, 0.5
            self.velocity_max, self.acceleration_max = 13.0, 10.0
        if class_id == 9: # Chair
            self.height_min, self.height_mean, self.height_max = 0.5, 1.0, 1.5
            self.width_min, self.width_mean, self.width_max = 0.5, 0.6, 0.7
            self.length_min, self.length_mean, self.length_max = 0.5, 0.6, 0.7
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 10: # Cow
            self.height_min, self.height_mean, self.height_max = 1.3, 1.3, 1.5
            self.width_min, self.width_mean, self.width_max = 0.5, 0.6, 0.7
            self.length_min, self.length_mean, self.length_max = 1.8, 2.3, 2.8
            self.velocity_max, self.acceleration_max = 10.0, 2.0
        if class_id == 11: # Dining table
            self.height_min, self.height_mean, self.height_max = 0.7, 0.75, 0.9
            self.width_min, self.width_mean, self.width_max = 0.5, 1.0, 1.5
            self.length_min, self.length_mean, self.length_max = 0.5, 2.0, 5.0
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 12: # Dog
            self.height_min, self.height_mean, self.height_max = 0.2, 0.35, 0.8
            self.width_min, self.width_mean, self.width_max = 0.15, 0.2, 0.3
            self.length_min, self.length_mean, self.length_max = 0.3, 0.6, 1.3
            self.velocity_max, self.acceleration_max = 13.0, 10.0
        if class_id == 13: # Horse
            self.height_min, self.height_mean, self.height_max = 1.2, 1.6, 1.8
            self.width_min, self.width_mean, self.width_max = 0.3, 0.45, 0.6
            self.length_min, self.length_mean, self.length_max = 2.0, 2.4, 3.0
            self.velocity_max, self.acceleration_max = 13.0, 4.0
        if class_id == 14: # Motorbike
            self.height_min, self.height_mean, self.height_max = 0.6, 1.1, 1.4
            self.width_min, self.width_mean, self.width_max = 0.4, 0.6, 0.8
            self.length_min, self.length_mean, self.length_max = 1.2, 1.5, 1.7
            self.velocity_max, self.acceleration_max = 50.0, 10.0
        if class_id == 15: # Person
            self.height_min, self.height_mean, self.height_max = 1.0, 1.7, 2.0
            self.width_min, self.width_mean, self.width_max = 0.5, 0.6, 0.7
            self.length_min, self.length_mean, self.length_max = 0.25, 0.35, 0.45
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 16: # Potted plant
            self.height_min, self.height_mean, self.height_max = 0.2, 0.35, 1.0
            self.width_min, self.width_mean, self.width_max = 0.15, 0.25, 0.4
            self.length_min, self.length_mean, self.length_max = 0.15, 0.25, 0.4
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 17: # Sheep
            self.height_min, self.height_mean, self.height_max = 0.8, 1.2, 1.4
            self.width_min, self.width_mean, self.width_max = 0.25, 0.35, 0.45
            self.length_min, self.length_mean, self.length_max = 1.0, 1.3, 1.5
            self.velocity_max, self.acceleration_max = 10.0, 6.0
        if class_id == 18: # Sofa
            self.height_min, self.height_mean, self.height_max = 0.7, 1.0, 1.4
            self.width_min, self.width_mean, self.width_max = 0.7, 1.0, 1.5
            self.length_min, self.length_mean, self.length_max = 1.5, 2.0, 4.0
            self.velocity_max, self.acceleration_max = 10.0, 3.0
        if class_id == 19: # Train
            self.height_min, self.height_mean, self.height_max = 2.5, 3.0, 5.0
            self.width_min, self.width_mean, self.width_max = 2.5, 3.1, 3.5
            self.length_min, self.length_mean, self.length_max = 25.0, 150.0, 1000.0
            self.velocity_max, self.acceleration_max = 50.0, 1.0
        if class_id == 20: # TV Monitor
            self.height_min, self.height_mean, self.height_max = 0.2, 0.5, 1.2
            self.width_min, self.width_mean, self.width_max = 0.3, 0.8, 2.0
            self.length_min, self.length_mean, self.length_max = 0.04, 0.1, 0.5
            self.velocity_max, self.acceleration_max = 10.0, 3.0

    def coordinates_from_pattern(self):
        """
        Calculates world coordinates from pattern center point, pattern height,
        camera parameters and default body height
        """
        sw = self.pattern.camera.sensor_width
        sh = self.pattern.camera.sensor_height
        pw = self.pattern.camera.image_width
        ph = self.pattern.camera.image_height
        hi = self.pattern.y_max - self.pattern.y_min
        h = self.height_mean
        f = self.pattern.camera.focal_length
        xp, yp = self.pattern.center_point()

        xc = -sw/2.0 + xp*sw/pw
        yc = sh/2.0 - yp*sh/ph
        zc = -f

        alfa = atan(yc/f)
        beta = atan(xc/f)

        d = f*h/(cos(alfa)*cos(beta)*hi*sh/ph)
        t = d / sqrt(xc**2.0 + yc**2.0 + zc**2.0)

        xo = t*xc
        yo = t*yc
        zo = t*zc

        return xo, yo, zo

#------------------------------------------------------------------------------
class BoundingBox:
    """
    Base class for detections and patterns
    """
    def __init__(self, x_min, x_max, y_min, y_max, image_width, image_height):
        """
        Initialization based on coordinates
        """
        # References
        # Attributes
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.border_left = 1
        self.border_right = 1
        self.border_top = 1
        self.border_bottom = 1
        self.set_border_behaviour(image_width, image_height)

    def border_count(self):
        """
        Counts the number of borders touching box
        """
        count = 0
        if self.border_left > 1:
            count += 1
        if self.border_right > 1:
            count += 1
        if self.border_top > 1:
            count += 1
        if self.border_bottom > 1:
            count += 1
        return count

    def center_point(self):
        """
        Calculate center point
        """
        x_center = (self.x_min + self.x_max) / 2.0
        y_center = (self.y_min + self.y_max) / 2.0
        return x_center, y_center

    def is_reliable(self):
        """
        Decides whether location is reliable, based on border behaviour
        """
        if self.border_left in (3, 5, 6):
            return False
        if self.border_right in (3, 5, 6):
            return False
        if self.border_top in (3, 5, 6):
            return False
        if self.border_bottom in (3, 5, 6):
            return False
        return True

    def is_vanished(self):
        """
        Checks if the box size goes zero or negative
        """
        c1 = (self.x_max <= self.x_min)
        c2 = (self.y_max <= self.y_min)
        if c1 or c2:
            return True
        return False

    def set_border_behaviour(self, image_width, image_height):
        """
        Sets the border information according to:
            1 = in normal area
            2 = in normal + border area
            3 = in normal + out of window area
            4 = inside border area
            5 = in border + out of window area
            6 = out of window area
        """
        # Left
        self.border_left = 1
        if self.x_max >= BORDER_WIDTH and \
           self.x_min <= BORDER_WIDTH and \
           self.x_min >= 0:
            self.border_left = 2
        if self.x_max >= BORDER_WIDTH and \
           self.x_min <= 0:
            self.border_left = 3
        if self.x_max <= BORDER_WIDTH and \
           self.x_min <= BORDER_WIDTH and \
           self.x_min >= 0:
            self.border_left = 4
        if self.x_max <= BORDER_WIDTH and \
           self.x_min <= 0:
            self.border_left = 5
        if self.x_min <= 0 and \
           self.x_max <= 0:
            self.border_left = 6
        # Right
        self.border_right = 1
        if self.x_min <= (image_width - BORDER_WIDTH) and \
           self.x_max >= (image_width - BORDER_WIDTH) and \
           self.x_max <= image_width:
            self.border_right = 2
        if self.x_min <= (image_width - BORDER_WIDTH) and \
           self.x_max >= image_width:
            self.border_right = 3
        if self.x_min >= (image_width - BORDER_WIDTH) and \
           self.x_max >= (image_width - BORDER_WIDTH) and \
           self.x_max <= image_width:
            self.border_right = 4
        if self.x_min >= (image_width - BORDER_WIDTH) and \
           self.x_max >= image_width:
            self.border_right = 5
        if self.x_min >= image_width and \
           self.x_max >= image_width:
            self.border_right = 6
        # Top
        self.border_top = 1
        if self.y_max >= BORDER_WIDTH and \
           self.y_min <= BORDER_WIDTH and \
            self.y_min >= 0:
            self.border_top = 2
        if self.y_max >= BORDER_WIDTH and \
           self.y_min <= 0:
            self.border_top = 3
        if self.y_max <= BORDER_WIDTH and \
            self.y_min <= BORDER_WIDTH and \
            self.y_min >= 0:
            self.border_top = 4
        if self.y_max <= BORDER_WIDTH and \
           self.y_min <= 0:
            self.border_top = 5
        if self.y_min <= 0 and \
           self.y_max <= 0:
            self.border_top = 6
        # Bottom
        self.border_bottom = 1
        if self.y_min <= (image_height - BORDER_WIDTH) and \
           self.y_max >= (image_height - BORDER_WIDTH) and \
           self.y_max <= image_height:
            self.border_bottom = 2
        if self.y_min <= (image_height - BORDER_WIDTH) and \
           self.y_max >= image_height:
            self.border_bottom = 3
        if self.y_min >= (image_height - BORDER_WIDTH) and \
           self.y_max >= (image_height - BORDER_WIDTH) and \
            self.y_max <= image_height:
            self.border_bottom = 4
        if self.y_min >= (image_height - BORDER_WIDTH) and \
           self.y_max >= image_height:
            self.border_bottom = 5
        if self.y_min >= image_height and \
           self.y_max >= image_height:
            self.border_bottom = 6

#------------------------------------------------------------------------------
class Camera:
    """
    Camera observing surrounding, located and oriented in world
    """
    def __init__(self, world, focal_length, sensor_width, sensor_height,
                 x, y, z, yaw, pitch, roll, videofile):
        """
        Initialization
        """
        # References
        self.world = world
        self.patterns = []
        # Attributes
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.field_of_view = 2.0*atan(sensor_width / (2.0*focal_length))
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.videofile = videofile
        self.video = cv2.VideoCapture(videofile)
        self.image_width = int(self.video.get(3))
        self.image_height = int(self.video.get(4))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        world.delta_time = 1.0 / self.fps
        self.current_frame = 0
        self.size_ratio = 900.0 / self.image_width
        frame = np.zeros((self.image_height, self.image_width, 3), np.uint8)
        frame = cv2.resize(frame, (0, 0), None, self.size_ratio, self.size_ratio)
        cv2.imshow(self.videofile, frame)
        cv2.moveWindow(self.videofile, 20, 20)

    def add_pattern(self, detection):
        """
        Create a new pattern based on detection, if appropriate
        """
        # Don't create if confidence too low
        if detection.confidence < CONFIDENFE_LEVEL_CREATE:
            return False

        # Don't create if detection is in border area or out of screen
        if detection.border_left > 3 or detection.border_right > 3 or \
            detection.border_top > 3 or detection.border_bottom > 3:
            return False

        # Don't create if touching 3 or more borders
        if detection.border_count() > 2:
            return False

        new_pattern = Pattern(self, detection)
        new_body = Body(self.world, new_pattern)
        new_pattern.body = new_body
        self.world.add_body(new_body)
        self.patterns.append(new_pattern)

        return True
    
    def close(self):
        """
        Release resources
        """
        self.video.release()
        cv2.destroyWindow(self.videofile)
    
    def detect(self, image, current_time):
        """
        Detection of objects based on MobileNet and SSD
        """
        (height, width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, \
                                     (300, 300), 127.5)
        # Pass the blob through the network and obtain the detections
        NET.setInput(blob)
        detections = NET.forward()
        objects = []
        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > CONFIDENFE_LEVEL_UPDATE:
                # Extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x_min, y_min, x_max, y_max) = box.astype("int")
                objects.append(Detection(time, class_id, x_min, x_max, y_min, 
                                         y_max, confidence, width, height))

        return objects

    def remove_pattern(self, pattern):
        """
        Remove pattern and clear references
        """
        self.patterns.remove(pattern)
        pattern.body.pattern = None

    def update(self, current_time, delta_time):
        """
        Detections are created and matched to previous patterns. New patterns
        are created if necessary and previous patters removed if not matched
        recently.
        """
        # Read in new frame
        ret, frame_video = self.video.read()
        if not ret:
            return False # end of video file
        self.current_frame += 1

        detections = self.detect(frame_video, current_time)

        # Process previous patterns
        if len(self.patterns) > 0:
            # Predict new coordinates for each pattern
            for pattern in self.patterns:
                pattern.predict(delta_time)
            # If the predicted location is out of screen, it will be removed
            removes = []
            for pattern in self.patterns:
                if pattern.border_left == 6 or pattern.border_right == 6 or \
                   pattern.border_top == 6 or pattern.border_bottom == 6:
                    removes.append(pattern)
            # Delete removed objects
            for remove in removes:
                self.remove_pattern(remove)

        # Match detections to patterns
        if len(self.patterns) > 0 and len(detections) > 0:
            # Calculate cost matrix for the Hungarian algorithm (assignment problem)
            cost = np.zeros((len(detections), len(self.patterns)))
            detection_index = 0
            for detection in detections:
                pattern_index = 0
                for pattern in self.patterns:
                    cost[detection_index, pattern_index] = \
                        detection.pattern_distance_with_class(pattern)
                    pattern_index += 1
                detection_index += 1
            # Remove rows and columns with no values inside SIMILARITY_DISTANCE
            count_detections = len(detections)
            count_patterns = len(self.patterns)
            remove_detections = []
            for detection_index in range(0, count_detections):
                found = False
                for pattern_index in range(0, count_patterns):
                    if cost[detection_index, pattern_index] < SIMILARITY_DISTANCE:
                        found = True
                if not found:
                    remove_detections.append(detections[detection_index])
            remove_patterns = []
            for pattern_index in range(0, count_patterns):
                found = False
                for detection_index in range(0, count_detections):
                    if cost[detection_index, pattern_index] < SIMILARITY_DISTANCE:
                        found = True
                if not found:
                    remove_patterns.append(self.patterns[pattern_index])
            detections_to_match = []
            patterns_to_match = []
            for detection in detections:
                if detection not in remove_detections:
                    detections_to_match.append(detection)
            for pattern in self.patterns:
                if pattern not in remove_patterns:
                    patterns_to_match.append(pattern)
            # The optimal assignment without far-away objects:
            cost_to_match = np.zeros((len(detections_to_match), \
                                      len(patterns_to_match)))
            row_index = -1
            row_index_original = 0
            for detection in detections:
                if detection not in remove_detections:
                    row_index += 1
                    col_index = -1
                    col_index_original = 0
                    for pattern in self.patterns:
                        if pattern not in remove_patterns:
                            col_index += 1
                            cost_to_match[row_index, col_index] = cost[row_index_original, col_index_original]
                        col_index_original += 1
                row_index_original += 1
            # Find the optimal assigment
            row_ind, col_ind = linear_sum_assignment(cost_to_match)
            # Update matched objects
            match_index = 0
            for row in row_ind:
                i1 = row_ind[match_index]
                i2 = col_ind[match_index]
                detection = detections_to_match[i1]
                pattern = patterns_to_match[i2]
                distance = cost_to_match[i1, i2]
                if distance > SIMILARITY_DISTANCE:
                    match_index += 1
                else:
                    pattern.correct(detection, delta_time)
                    pattern.detected = True
                    pattern.retention_count = 0
                    pattern.detections.append(detection)
                    detection.matched = True
                    match_index += 1

            # If the corrected location is out of screen, it will be removed
            removes = []
            for pattern in self.patterns:
                if pattern.border_left == 6 or pattern.border_right == 6 \
                or pattern.border_top == 6 or pattern.border_bottom == 6:
                    removes.append(pattern)

            # Delete removed patterns
            for remove in removes:
                self.remove_pattern(remove)

        # Remove vanished patterns
        removes = []
        for pattern in self.patterns:
            if pattern.is_vanished():
                removes.append(pattern)

        # Delete removed patterns
        for remove in removes:
            self.remove_pattern(remove)

        # Remove patterns touching 3 or more borders
        removes = []
        for pattern in self.patterns:
            if pattern.border_count() > 2:
                removes.append(pattern)

        # Delete removed patterns
        for remove in removes:
            self.remove_image_object(remove)

        # Any pattern not matched is changed to not detected state
        for pattern in self.patterns:
            if not pattern.matched:
                pattern.detected = False
                pattern.retention_count += 1

        # Remove patterns that has not been detected for some time
        removes = []
        for pattern in self.patterns:
            if pattern.retention_count > RETENTION_COUNT_MAX:
                removes.append(pattern)

        # Delete removed patterns
        for remove in removes:
            self.remove_pattern(remove)

        # If no match for a detection, create a new pattern
        for detection in detections:
            if not detection.matched:
                found = False
                for pattern in self.patterns:
                    if detection.pattern_distance(pattern) < SIMILARITY_DISTANCE:
                        found = True
                if not found: # Only if there is no other pattern near
                    self.add_pattern(detection)

        # Draw heading
        label = "Time {0:<.2f}, frame {1:d}".format(current_time, self.current_frame)
        cv2.putText(frame_video, label, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

        # Draw detections
        for detection in detections:
            cv2.rectangle(frame_video, (detection.x_min, detection.y_min), \
                          (detection.x_max, detection.y_max), (255,255,255), 1)

        # Draw patterns
        for pattern in self.patterns:
            cv2.rectangle(frame_video, (int(pattern.x_min), int(pattern.y_min)), 
                          (int(pattern.x_max), int(pattern.y_max)), 
                          pattern.bounding_box_color, 2)
            label = "{0:s}: {1:.2f}".format(CLASS_NAMES[pattern.class_id],
                     pattern.confidence)
            ytext = int(pattern.y_min) - 15 if int(pattern.y_min) - 15 > 15 \
                else int(pattern.y_min) + 15
            cv2.putText(frame_video, label, (int(pattern.x_min), ytext), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if (pattern.is_reliable()):
                x_center, y_center = pattern.center_point()
                x_variance, y_variance = pattern.location_variance()
                x_std2 = 2.0 * sqrt(x_variance)
                y_std2 = 2.0 * sqrt(y_variance)
                cv2.ellipse(frame_video, (int(x_center), int(y_center)), 
                            (int(x_std2),int(y_std2)), 0.0, 0, 360, 
                            pattern.bounding_box_color, 2)
                x_center_velocity, y_center_velocity = pattern.velocity()
                cv2.arrowedLine(frame_video, (int(x_center), int(y_center)), 
                                (int(x_center+x_center_velocity), 
                                 int(y_center+y_center_velocity)), 
                                 pattern.bounding_box_color, 2)

        # Resize frame and display
        frame_display = cv2.resize(frame_video, (0, 0), None, self.size_ratio, 
                           self.size_ratio)
        cv2.imshow(self.videofile, frame_display)
        cv2.waitKey(1)

        return True
#------------------------------------------------------------------------------
class Detection(BoundingBox):
    """
    Output from object detector, measurement
    """
    def __init__(self, current_time, class_id, x_min, x_max, y_min, y_max, 
                 confidence, image_width, image_height):
        """
        Initialization
        """
        super().__init__(x_min, x_max, y_min, y_max, image_width, image_height)
        # References
        self.pattern = None
        # Attributes
        self.time = current_time
        self.class_id = class_id
        self.confidence = confidence
        self.matched = False

    def pattern_distance(self, pattern):
        """
        Calculates distance to a persistent pattern
        """
        dx_min = abs(self.x_min - pattern.x_min)
        dx_max = abs(self.x_max - pattern.x_max)
        dy_min = abs(self.y_min - pattern.y_min)
        dy_max = abs(self.y_max - pattern.y_max)
        #
        horizontal_size = max(pattern.x_max - pattern.x_min, 1)
        vertical_size = max(pattern.y_max - pattern.y_min, 1)
        #
        dx_min /= horizontal_size
        dx_max /= horizontal_size
        dy_min /= vertical_size
        dy_max /= vertical_size
        #
        d = (dx_min + dx_max + dy_min + dy_max) / 4.0
        return d

    def pattern_distance_with_class(self, pattern):
        """
        Calculates distance to persistent pattern penalizing class difference
        """
        d = self.pattern_distance(pattern)
        if self.class_id != pattern.class_id:
            d += 100.0
        return d

#------------------------------------------------------------------------------
class Event:
    """
    World event of interest
    """
    def __init__(self, speechsynthesizer, time, priority, text):
        """
        Initialization
        Priority: 0 highest
        """
        # References
        self.speechsynthesizer = speechsynthesizer
        # Attributes
        self.time = time
        self.priority = priority
        self.text = text

#------------------------------------------------------------------------------
class Pattern(BoundingBox):
    """
    Persistent pattern in image
    """
    def __init__(self, camera, detection):
        """
        Initialization based on Detection
        """
        super().__init__(detection.x_min, detection.x_max, detection.y_min,
                         detection.y_max, camera.image_width, 
                         camera.image_height)
        # References
        self.camera = camera
        self.detections = []
        self.add_detection(detection)
        self.body = None
        # Attributes
        self.class_id = detection.class_id
        self.vx_min = 0.0
        self.vx_max = 0.0
        self.vy_min = 0.0
        self.vy_max = 0.0
        self.sigma_x_min = np.array([[PATTERN_ALFA, 0],
                                     [0.0, PATTERN_BETA]])
        self.sigma_x_max = np.array([[PATTERN_ALFA, 0],
                                     [0.0, PATTERN_BETA]])
        self.sigma_y_min = np.array([[PATTERN_ALFA, 0],
                                     [0.0, PATTERN_BETA]])
        self.sigma_y_max = np.array([[PATTERN_ALFA, 0],
                                     [0.0, PATTERN_BETA]])
        self.confidence = detection.confidence
        self.bounding_box_color = (rnd.randint(0,255), rnd.randint(0,255), 
                                   rnd.randint(0,255))
        self.retention_count = 0
        self.matched = False

    def add_detection(self, detection):
        """
        Add detection to list of detections
        """
        self.detections.append(detection)

    def velocity(self):
        """
        Calculate center point velocity
        """
        x_center = (self.vx_min + self.vx_max) / 2.0
        y_center = (self.vy_min + self.vy_max) / 2.0
        return x_center, y_center

    def correct(self, detection, delta):
        """
        Correct bounding box coordinates based on new detection and the state
        delta seconds ago. Also updates other attribute information.
        """
        self.confidence = detection.confidence
        a = np.array([[1.0, delta], [0.0, 1.0]])
        #
        k_x_min = self.sigma_x_min.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_x_min).dot(PATTERN_C.T) + PATTERN_Q))
        mu_xmin = a.dot(np.array([[self.x_min], [self.vx_min]]))
        mu_xmin = mu_xmin + k_x_min.dot(detection.x_min - PATTERN_C.dot(mu_xmin))
        self.x_min = mu_xmin[0, 0]
        self.vx_min = mu_xmin[1, 0]
        self.sigma_x_min = (np.eye(2)-k_x_min.dot(PATTERN_C)).dot(self.sigma_x_min)
        #
        k_x_max = self.sigma_x_max.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_x_max).dot(PATTERN_C.T) + PATTERN_Q))
        mu_xmax = a.dot(np.array([[self.x_max], [self.vx_max]]))
        mu_xmax = mu_xmax + k_x_max.dot(detection.x_max - PATTERN_C.dot(mu_xmax))
        self.x_max = mu_xmax[0, 0]
        self.vx_max = mu_xmax[1, 0]
        self.sigma_x_max = (np.eye(2)-k_x_max.dot(PATTERN_C)).dot(self.sigma_x_max)
        #
        k_y_min = self.sigma_y_min.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_y_min).dot(PATTERN_C.T) + PATTERN_Q))
        mu_ymin = a.dot(np.array([[self.y_min], [self.vy_min]]))
        mu_ymin = mu_ymin + k_y_min.dot(detection.y_min - PATTERN_C.dot(mu_ymin))
        self.y_min = mu_ymin[0, 0]
        self.vy_min = mu_ymin[1, 0]
        self.sigma_y_min = (np.eye(2)-k_y_min.dot(PATTERN_C)).dot(self.sigma_y_min)
        #
        k_y_max = self.sigma_y_max.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_y_max).dot(PATTERN_C.T) + PATTERN_Q))
        mu_ymax = a.dot(np.array([[self.y_max], [self.vy_max]]))
        mu_ymax = mu_ymax + k_y_max.dot(detection.y_max - PATTERN_C.dot(mu_ymax))
        self.y_max = mu_ymax[0, 0]
        self.vy_max = mu_ymax[1, 0]
        self.sigma_y_max = (np.eye(2)-k_y_max.dot(PATTERN_C)).dot(self.sigma_y_max)
        #
        self.set_border_behaviour(self.camera.image_width, self.camera.image_height)
        self.matched = True

    def location_variance(self):
        """
        Calculate location variance by summing covariance matrix diagonal
        elements
        """
        x_variance = self.sigma_x_min[0, 0] + self.sigma_x_max[0, 0]
        y_variance = self.sigma_y_min[0, 0] + self.sigma_y_max[0, 0]
        return x_variance, y_variance

    def predict(self, delta):
        """
        Predicts bounding box coordinates delta seconds ahead, based on Kalman
        filtering
        """
        a = np.array([[1.0, delta], [0.0, 1.0]])
        #
        mu_xmin = a.dot(np.array([[self.x_min], [self.vx_min]]))
        self.x_min = mu_xmin[0, 0]
        self.vx_min = mu_xmin[1, 0]
        self.sigma_x_min = a.dot(self.sigma_x_min).dot(a.T) + PATTERN_R
        #
        mu_xmax = a.dot(np.array([[self.x_max], [self.vx_max]]))
        self.x_max = mu_xmax[0, 0]
        self.vx_max = mu_xmax[1, 0]
        self.sigma_x_max = a.dot(self.sigma_x_max).dot(a.T) + PATTERN_R
        #
        mu_ymin = a.dot(np.array([[self.y_min], [self.vy_min]]))
        self.y_min = mu_ymin[0, 0]
        self.vy_min = mu_ymin[1, 0]
        self.sigma_y_min = a.dot(self.sigma_y_min).dot(a.T) + PATTERN_R
        #
        mu_ymax = a.dot(np.array([[self.y_max], [self.vy_max]]))
        self.y_max = mu_ymax[0, 0]
        self.vy_max = mu_ymax[1, 0]
        self.sigma_y_max = a.dot(self.sigma_y_max).dot(a.T) + PATTERN_R
        #
        self.set_border_behaviour(self.camera.image_width, 
                                  self.camera.image_height)
        self.matched = False

#------------------------------------------------------------------------------
class Presentation:
    """
    Class for presenting world information in human understandable form
    """
    def __init__(self, world):
        """
        Initialization
        """
        # References
        self.world = world
        # Attributes

    def close(self):
        """
        Release resources
        """
        pass

    def update(self, current_time):
        """
        Update presentation at time t
        """
        pass

#------------------------------------------------------------------------------
class PresentationMap(Presentation):
    """
    World event of interest
    """
    def __init__(self, world, map_id, height_pixels, width_pixels):
        """
        Initialization
        """
        super().__init__(world)
        self.map_id =map_id
        self.height_pixels = height_pixels
        self.width_pixels = width_pixels
        self.frame = np.zeros((height_pixels, width_pixels, 3), np.uint8)
        self.window_name = "Map " + str(map_id)
        cv2.imshow(self.window_name, self.frame)
        cv2.moveWindow(self.window_name, 940, 20)
        
    def close(self):
        """
        Release resources
        """
        cv2.destroyWindow(self.window_name)

    def update(self, current_time):
        """
        Update presentation at time t
        """
        # No bodies, now update
        if len(self.world.bodies) == 0:
            return
        self.frame = np.zeros((self.height_pixels, self.width_pixels, 3), 
                              np.uint8)
        # Get the min and max coordinates
        c_max = -inf
        for body in self.world.bodies:
            if abs(body.x) > c_max:
                c_max = abs(body.x)
            if abs(body.z) > c_max:
                c_max = abs(body.z)
        extension_meters = max(c_max+10.0, 10.0) # At least 10 meters
        extension_pixels = min(self.height_pixels, self.width_pixels)
        pixels_meter = 0.5*extension_pixels / extension_meters
        # Draw
        for body in self.world.bodies:
            xc = self.width_pixels/2.0 + body.x * pixels_meter
            yc = self.height_pixels/2.0 + body.z * pixels_meter
            cv2.ellipse(self.frame, (int(xc), int(yc)), (1,1), 0.0, 0, 360, 
                        (255,255,255), 2)
        # Radar circles
        radius = 10.0
        while radius < extension_meters:
            radius_pixels = int(radius * pixels_meter)
            cv2.circle(self.frame, (int(self.width_pixels/2), 
                                    int(self.height_pixels/2)), radius_pixels, 
                                    (255,255,255), 1)
            radius += 10.0
        # Camera field of views
        for camera in self.world.cameras:
            x = self.height_pixels / 2.0 * tan(camera.field_of_view/2.0)
            pt1 = (int(self.width_pixels/2), int(self.height_pixels/2))
            pt2 = (int(self.width_pixels/2-x),0)
            cv2.line(self.frame, pt1, pt2, (255,255,255), 1)
            pt2 = (int(self.width_pixels/2+x),0)
            cv2.line(self.frame, pt1, pt2, (255,255,255), 1)
        
        cv2.imshow(self.window_name, self.frame)
                
#------------------------------------------------------------------------------
class SpeechSynthesizer:
    """
    Class for spelling out text. Relies on pyttsc3 package and currently
    waits until the talk is over.
    """
    def __init__(self, world):
        """
        Initialization
        """
        # References
        self.world = world
        self.engine = pyttsx3.init()
        self.events = []
        # Attributes

    def say(self, text):
        """
        Say the text
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def add_event(self, time, category, priority, text):
        """
        Create a new event to be spelled out
        """
        new_event = Event(self, time, category, priority, text)
        self.events.append(new_event)
        self.say(text)

#------------------------------------------------------------------------------
class Trace:
    """
    History of body movement and other body changes
    """
    def __init__(self, current_time, body):
        """
        Initialization
        """
        # References
        self.body = body
        # Attributes
        self.time = current_time
        self.class_id = body.class_id
        self.x = body.x
        self.y = body.y
        self.z = body.z
        self.vx = body.vx
        self.vy = body.vy
        self.vz = body.vz
        self.ax = body.ax
        self.ay = body.ay
        self.az = body.az

#------------------------------------------------------------------------------
class World:
    """
    3 dimensional model of the physical world
    """
    def __init__(self):
        """
        Initialization
        """
        # References
        self.bodies = []
        self.cameras = []
        self.presentations = []
        self.speech_synthesizer = SpeechSynthesizer(self)
        # Attributes
        self.current_time = 0.0
        self.delta_time = 1.0/30.0

    def add_body(self, body):
        """
        Add a body
        """
        self.bodies.append(body)

    def add_camera(self, camera):
        """
        Adds a camera
        """
        self.cameras.append(camera)
        
    def add_presentation(self, presentation):
        """
        Adds a presentation
        """
        self.presentations.append(presentation)
        
    def close(self):
        """
        Release resources
        """
        for camera in self.cameras:
            camera.close()
        for presentation in self.presentations:
            presentation.close()

    def run(self):
        """
        Run the world
        """
        more = True
        while more:
            more = self.update()
        self.close()
        
    def update(self):
        """
        Each camera is asked to update their patterns. Patterns are projected
        into bodies. This is repeated until cameras have no more frames.
        """
        more = True
        while more:
            more = True
            for camera in self.cameras:
                more_camera = camera.update(self.current_time, self.delta_time)
                if not more_camera:
                    more = False
            for body in self.bodies:
                if body.pattern is not None:
                    xo, yo, zo = body.coordinates_from_pattern()
                    body.x = xo
                    body.y = yo
                    body.z = zo
            for presentation in self.presentations:
                presentation.update(self.current_time)
            self.current_time += self.delta_time

#------------------------------------------------------------------------------
TEST_VIDEOS = ['videos/AWomanStandsOnTheSeashore-10058.mp4', # 0
               'videos/BlueTit2975.mp4', # 1
               'videos/Boat-10876.mp4', # 2
               'videos/Calf-2679.mp4', # 3
               'videos/Cars133.mp4', # 4
               'videos/CarsOnHighway001.mp4', # 5
               'videos/Cat-3740.mp4', # 6
               'videos/Dog-4028.mp4', # 7
               'videos/Dunes-7238.mp4', # 8
               'videos/Hiker1010.mp4', # 9
               'videos/Horse-2980.mp4', # 10
               'videos/Railway-4106.mp4', # 11
               'videos/SailingBoat6415.mp4', # 12
               'videos/Sheep-12727.mp4', # 13
               'videos/Sofa-11294.mp4'] # 14

def run_application():
    """
    Example application
    """
    world = World()
    camera = Camera(world, focal_length=0.050, sensor_width=0.0359, 
                     sensor_height=0.0240, x=0.0, y=0.0, z=0.0, yaw=0.0, 
                     pitch=0.0, roll=0.0, videofile=TEST_VIDEOS[5])
    world.add_camera(camera)

    presentationmap= PresentationMap(world, 1, 500, 500)
    world.add_presentation(presentationmap)

    world.run()

if __name__ == "__main__":
    run_application()