# -*- coding: utf-8 -*-
"""
ShadowWorld version 2.1
Created on Mon Jan 29 08:30:49 2018

@author: Sakari Lampola
"""
# Imports----------------------------------------------------------------------
import numpy as np
import cv2
import time
import pandas as pd
import random as rnd
import pyttsx3
import winsound
from math import atan, cos, sqrt, tan, exp, log
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
import pykitti
import sys
import tensorflow as tf
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import matplotlib as plt

# Hyperparameters--------------------------------------------------------------
BODY_ALFA = 100000.0 # Body initial location error variance
BODY_BETA = 100000.0 # Body initial velocity error variance
BODY_C = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                  ]) # Body measurement matrix
BODY_DATA_COLLECTION_COUNT = 30 # How many frames until notification
BODY_Q = np.array([[200.0, 0.0,   0.0],
                   [0.0, 200.0,   0.0],
                   [0.0,   0.0, 200.0]]) # Body measurement variance 200
BODY_R = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                  ]) # Body state equation covariance
#
BORDER_WIDTH = 30 # Part of video window for special pattern behaviour
#
COLLISION_SAMPLES = 1000 # How many samples are drawn for the collision detection
COLLISION_MIN_LEVEL = 0.01 # Smallest collision level to be spelled out
#
CONFIDENFE_LEVEL_CREATE = 0.80 # How confident must be to create new pattern
CONFIDENFE_LEVEL_UPDATE = 0.20 # How confident must be to update pattern
#
FORECAST_DELTA = 0.01 # Time step for the forecast
FORECAST_A = np.array([[1.0, 0.0, 0.0, FORECAST_DELTA,            0.0,            0.0],
                       [0.0, 1.0, 0.0,            0.0, FORECAST_DELTA,            0.0],
                       [0.0, 0.0, 1.0,            0.0,            0.0, FORECAST_DELTA],
                       [0.0, 0.0, 0.0,            1.0,            0.0,            0.0],
                       [0.0, 0.0, 0.0,            0.0,            1.0,            0.0],
                       [0.0, 0.0, 0.0,            0.0,            0.0,           1.0]])
FORECAST_COUNT = 500 # How many time steps ahead a body forecast is made
FORECAST_INTERVAL = 1.0 # How often forecast is made
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = r'D:\Thesis\Models\faster_rcnn_resnet101_kitti_2018_01_28' + '/frozen_inference_graph.pb'
#
PATTERN_ALFA = 200.0 # Pattern initial location error variance
PATTERN_BETA = 10000.0 # Pattern initial velocity error variance
PATTERN_C = np.array([[1.0, 0.0]]) # Pattern measurement matrix
PATTERN_Q = np.array([200.0]) # Pattern measurement variance
PATTERN_R = np.array([[0.1, 0.0],
                      [0.0, 1.0]]) # Pattern state equation covariance
#
RETENTION_COUNT_MAX = 30 # How many frames pattern is kept untedected
#
SIMILARITY_DISTANCE = 0.4 # Max distance for detection-pattern similarity
STEREO_MATCH_MEAN = pd.read_pickle("stereo/stereomatch_mean.pkl").as_matrix()
STEREO_MATCH_COVARIANCE = pd.read_pickle("stereo/stereomatch_covariance.pkl").as_matrix()
STEREO_MAX_DISTANCE = 20.0 # Max distance to stereo vision to be accurate
STEREO_MIN_LOG_PROBABILITY = 9 # Mimimun negative log probability for matching
# Other constants--------------------------------------------------------------
CLASS_NAMES = ["background", "car", "person"]
MAP_EXTENT = 101.0
SENSOR_WIDTH = 1392
SENSOR_HEIGHT = 512
UI_VIDEO_SCALE = 0.8
UI_X0 = 0
UI_X1 = 1114
UI_X2 = 2037
UI_X3 = 2400
UI_Y0 = 0
UI_Y1 = 410
UI_Y2 = 820
UI_Y3 = 923
UI_Y4 = 1230
# Classes----------------------------------------------------------------------
class Body:
    """
    A physical entity in the world
    """
    def __init__(self, world, pattern_left, pattern_right):
        """
        Initialization
        """
        # References
        self.world = world
        self.pattern_left = pattern_left
        self.pattern_right = pattern_right
        self.events = []
        self.traces = []
        self.forecast = None
        # Attributes
        self.set_class_attributes(pattern_left.class_id)
        self.frame_age = 0
        self.x, self.y, self.z = self.coordinates_from_patterns()
        self.status = "measured"
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.sigma = np.array([[BODY_ALFA, 0.0,       0.0,       0.0,       0.0,       0.0],
                               [0.0, BODY_ALFA,       0.0,       0.0,       0.0,       0.0],
                               [0.0,       0.0, BODY_ALFA,       0.0,       0.0,       0.0],
                               [0.0,       0.0,       0.0, BODY_BETA,       0.0,       0.0],
                               [0.0,       0.0,       0.0,       0.0, BODY_BETA,       0.0],
                               [0.0,       0.0,       0.0,       0.0,       0.0, BODY_BETA]])
        self.forecast_color = pattern_left.bounding_box_color
        self.collision_probability = 0.0
        self.collision_probability_max = 0.0

    def add_trace(self, time):
        """
        Add a history trace
        """
        self.traces.append(Trace(time, self))
    
    def coordinates_from_patterns(self):
        """
        Calculates world coordinates from pattern center point, pattern height,
        camera parameters and default body height
        """
        if (self.pattern_left is None) or (self.pattern_left is None):
            return 0.0, 0.0, 0.0
        
        sw = self.pattern_left.camera.sensor_width
        sh = self.pattern_left.camera.sensor_height
        pw = self.pattern_left.camera.sensor_width_pixels
        ph = self.pattern_left.camera.sensor_height_pixels
        ri = self.pattern_left.radius()
        r = self.mean_radius()
        f = self.pattern_left.camera.focal_length
        xp, yp = self.pattern_left.center_point()

        xc = -sw/2.0 + xp*sw/pw
        yc = sh/2.0 - yp*sh/ph
        zc = -f

        alfa = atan(yc/f)
        beta = atan(xc/f)

        d_size = f*r/(cos(alfa)*cos(beta)*ri*sh/ph)
        
        disparity = (self.pattern_left.x_min+self.pattern_left.x_max)*0.5       
        disparity -= (self.pattern_right.x_min+self.pattern_right.x_max)*0.5       
        base = self.pattern_right.camera.x - self.pattern_left.camera.x

        if disparity > 0:
            d_stereo = f*base/(cos(alfa)*cos(beta)*disparity*sw/pw)
        else:
            d_stereo = d_size
        
        estimated_distance = (d_stereo + d_size)*0.5
        fraction = 0.25
        if estimated_distance < (1-fraction) * STEREO_MAX_DISTANCE:
            k_size = 0.0
            k_stereo = 1.0
        elif estimated_distance > (1+fraction) * STEREO_MAX_DISTANCE:
            k_size = 1.0
            k_stereo = 0.0
        else:
            l1 = estimated_distance - (1-fraction) * STEREO_MAX_DISTANCE
            l2 = (1+fraction)*STEREO_MAX_DISTANCE - (1-fraction)*STEREO_MAX_DISTANCE
            k_size = l1/l2
            k_stereo = 1 - k_size
        
        d = k_size* d_size + k_stereo * d_stereo
        t = d / sqrt(xc**2.0 + yc**2.0 + zc**2.0)

        xo = t*xc
        yo = t*yc
        zo = t*zc

        return xo, yo, zo

    def correct(self, x, y, z, delta):
        """
        Correct body location based on new measurement. 
        """
        self.status = "measured"
        measurement = np.array([[x], [y], [z]])
        #
        k = self.sigma.dot(BODY_C.T).dot(np.linalg.inv(BODY_C.dot(self.sigma).dot(BODY_C.T) + BODY_Q))
        mu = np.array([[self.x], 
                       [self.y], 
                       [self.z],
                       [self.vx], 
                       [self.vy], 
                       [self.vz]])
        mu = mu + k.dot(measurement - BODY_C.dot(mu))
        self.x  = mu[0, 0]
        self.y  = mu[1, 0]
        self.z  = mu[2, 0]
        self.vx = mu[3, 0]
        self.vy = mu[4, 0]
        self.vz = mu[5, 0]
        self.sigma = (np.eye(6)-k.dot(BODY_C)).dot(self.sigma)

    def detect_collision(self):
        """
        Calculate collision probability with the observer
        """
        if self.forecast is None:
            self.collision_probability = 0.0
            return
        
        mu = np.array([self.forecast.x_min_distance,
                       self.forecast.y_min_distance,
                       self.forecast.z_min_distance])
        sigma = self.forecast.sigma_min_distance[0:3,0:3]
        samples_location = np.random.multivariate_normal(mu, sigma, 
                                                         COLLISION_SAMPLES)
        samples_body_radius = np.random.lognormal(self.diameter_mu, 
                                                  self.diameter_sigma, 
                                                  COLLISION_SAMPLES)/2.0
        
        samples_observer_radius = np.random.lognormal(log(1.67), 
                                                      0.07, 
                                                      COLLISION_SAMPLES)/2.0 
                                                      
        count_collision = 0
        for i in range (COLLISION_SAMPLES):
            if np.linalg.norm(samples_location[i]) < (samples_observer_radius[i] +
                             samples_body_radius[i]):
                count_collision += 1
                
        self.collision_probability = count_collision / COLLISION_SAMPLES

    def make_forecast(self, time):
        """
        Make a forcast of body movement and uncertanties
        """
        self.forecast = Forecast(time, self)

    def location_variance(self):
        """
        Calculate location variance by summing covariance matrix diagonal
        elements
        """
        variance = self.sigma[0, 0] + self.sigma[1, 1]+ self.sigma[2, 2]
        return variance
    
    def mean_radius(self):
        """
        Calculate mean radius
        """
        mu = self.diameter_mu
        sigma = self.diameter_sigma
        return exp(mu + sigma * sigma / 2.0)/2.0

    def predict(self, delta):
        """
        Predicts body location, based on Kalman filtering.
        """
        self.status = "predicted"
        self.frame_age += 1
        a = np.array([[1.0, 0.0, 0.0, delta,   0.0,   0.0],
                      [0.0, 1.0, 0.0,   0.0, delta,   0.0],
                      [0.0, 0.0, 1.0,   0.0,   0.0, delta],
                      [0.0, 0.0, 0.0,   1.0,   0.0,   0.0],
                      [0.0, 0.0, 0.0,   0.0,   1.0,   0.0],
                      [0.0, 0.0, 0.0,   0.0,   0.0,   1.0]])
        #
        mu = a.dot(np.array([[self.x], 
                             [self.y], 
                             [self.z],
                             [self.vx], 
                             [self.vy], 
                             [self.vz]]))

        self.x  = mu[0, 0]
        self.y  = mu[1, 0]
        self.z  = mu[2, 0]
        self.vx = mu[3, 0]
        self.vy = mu[4, 0]
        self.vz = mu[5, 0]

        self.sigma = a.dot(self.sigma).dot(a.T) + BODY_R

    def set_class_attributes(self, class_id):
        """
        Sets class specific attributes
        """
        self.class_id = class_id
        if class_id == 1: # Car
            self.diameter_mu = log(3.6)
            self.diameter_sigma = 0.25
            self.velocity_max = 50.0
        if class_id == 2: # Person
            self.diameter_mu = log(1.67)
            self.diameter_sigma = 0.07
            self.velocity_max = 10.0

    def speed(self):
        return sqrt(self.vx**2.0 + self.vy**2.0 + self.vz**2.0)

#------------------------------------------------------------------------------
class BoundingBox:
    """
    Base class for detections and patterns
    """
    def __init__(self, x_min, x_max, y_min, y_max, image_width, image_height,
                 patch, confidence):
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
        self.confidence = confidence
        self.x = (x_min + x_max) / 2
        self.y = (y_min + y_max) / 2
        self.width = x_max - x_min
        self.height = y_max - y_min
        patch_hsv = plt.colors.rgb_to_hsv(patch)
        histo, bins = np.histogram(patch_hsv[:,:,0], 3)
        histo = histo / sum(histo)
        self.hue0 = histo[0]
        self.hue1 = histo[1]
        self.hue2 = histo[2]
        self.saturation = np.mean(patch_hsv[:,:,1])
        self.value = np.mean(patch_hsv[:,:,2])
        self.patch = patch

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

    def radius(self):
        """
        Calculate the radius of enclosing circle
        """
        h = (self.y_max - self.y_min)/2.0
        w = (self.x_max - self.x_min)/2.0
        if h<=0:
            return w
        if w<=0:
            return h
        return sqrt(h*h+w*w)

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
    def __init__(self, world, name, focal_length, sensor_width, sensor_height,
                 sensor_width_pixels, sensor_height_pixels, 
                 x, y, z, yaw, pitch, roll, frames):
        """
        Initialization
        """
        # References
        self.world = world
        self.patterns = []
        self.detections = []
        # Attributes
        self.name = name
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.sensor_width_pixels = sensor_width_pixels
        self.sensor_height_pixels = sensor_height_pixels
        self.image_width_pixels = sensor_width_pixels
        self.image_height_pixels = sensor_height_pixels
        self.field_of_view = 2.0*atan(sensor_width / (2.0*focal_length))
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.frames = frames
        self.frame_factory = iter(self.frames)
        self.current_frame_count = 0
        self.current_frame = None

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

        # Don't create if unreliable
        if not detection.is_reliable():
            return False

        # Don't create if vanished
        if detection.is_vanished():
            return False

        new_pattern = Pattern(self, detection)

        self.patterns.append(new_pattern)
        self.world.add_event(Event(self.world, self.world.current_time, 2,
                                   id(new_pattern), "Pattern created"))

        return True
    
    def close(self):
        """
        Release resources
        """
        pass
    
    def detect(self, image, current_time):
        """
        Detection of objects based on World object detector
        """
        (height, width) = image.shape[:2]
        objects = []
        output_dict = self.world.object_detector.detect(image)        
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        
        xi = int((self.sensor_width_pixels - width)/2)
        yi = int((self.sensor_height_pixels - height)/2)
        
        for i in range(boxes.shape[0]):
            if scores[i] >= CONFIDENFE_LEVEL_UPDATE:
                x_min = int(width*boxes[i,1]) + xi
                x_max = int(width*boxes[i,3]) + xi
                y_min = int(height*boxes[i,0]) + yi
                y_max = int(height*boxes[i,2]) + yi
                objects.append(Detection(time, classes[i], x_min, x_max,
                                         y_min, y_max, scores[i], 
                                         width, height, image[y_min:y_max,
                                                              x_min:x_max,:]))
        

        return objects

    def remove_pattern(self, pattern):
        """
        Remove pattern and clear references
        """
        self.world.add_event(Event(self.world, self.world.current_time, 2,
                                   id(pattern), "Pattern removed"))
        self.patterns.remove(pattern)
        if pattern.body is not None:
            pattern.body.pattern_left = None
            pattern.body.pattern_right = None

    def update(self, current_time, delta_time):
        """
        Detections are created and matched to previous patterns. New patterns
        are created if necessary and previous patters removed if not matched
        recently.
        """
        # Read in new frame
        try:
            frame_video_in = next(self.frame_factory)
        except StopIteration:
            return False, None # end of video file

        frame_video = frame_video_in.copy()
        self.image_height_pixels, self.image_width_pixels  = frame_video.shape[:2]

        self.current_frame = frame_video.copy()
        self.current_frame_count += 1

        self.detections = self.detect(frame_video, current_time)
        for detection in self.detections:
            self.world.add_event(Event(self.world, current_time, 3,
                                       id(detection), "Detection created"))

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
        if len(self.patterns) > 0 and len(self.detections) > 0:
            # Calculate cost matrix for the Hungarian algorithm (assignment problem)
            cost = np.zeros((len(self.detections), len(self.patterns)))
            detection_index = 0
            for detection in self.detections:
                pattern_index = 0
                for pattern in self.patterns:
                    cost[detection_index, pattern_index] = \
                        detection.pattern_distance_with_class(pattern)
                    pattern_index += 1
                detection_index += 1
            # Remove rows and columns with no values inside SIMILARITY_DISTANCE
            count_detections = len(self.detections)
            count_patterns = len(self.patterns)
            remove_detections = []
            for detection_index in range(0, count_detections):
                found = False
                for pattern_index in range(0, count_patterns):
                    if cost[detection_index, pattern_index] < SIMILARITY_DISTANCE:
                        found = True
                if not found:
                    remove_detections.append(self.detections[detection_index])
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
            for detection in self.detections:
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
            for detection in self.detections:
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
            self.remove_pattern(remove)

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
        for detection in self.detections:
            if not detection.matched:
                found = False
                for pattern in self.patterns:
                    if detection.pattern_distance(pattern) < SIMILARITY_DISTANCE:
                        found = True
                if not found: # Only if there is no other pattern near
                    self.add_pattern(detection)

        # Create and gather UI frame
        frame_out = np.zeros((int(SENSOR_HEIGHT), int(SENSOR_WIDTH), 3), np.uint8)
        xi = int((self.sensor_width_pixels - self.image_width_pixels)/2)
        yi = int((self.sensor_height_pixels - self.image_height_pixels)/2)
        iw = self.image_width_pixels
        ih = self.image_height_pixels
        frame_out[yi:yi+ih, xi:xi+iw, :] = frame_video

        # Draw detections
        for detection in self.detections:
            cv2.rectangle(frame_out, (detection.x_min, detection.y_min), \
                          (detection.x_max, detection.y_max), (255,255,255), 1)

        # Draw patterns
        for pattern in self.patterns:
            cv2.rectangle(frame_out, (int(pattern.x_min), int(pattern.y_min)), 
                          (int(pattern.x_max), int(pattern.y_max)), 
                          pattern.bounding_box_color, 1)
            x_center, y_center = pattern.center_point()
            cv2.circle(frame_out, (int(x_center), int(y_center)),
                       int(pattern.radius()), (255, 255, 255), 1)
            label = "{0:s}: {1:.2f}".format(CLASS_NAMES[pattern.class_id],
                     pattern.confidence)
            ytext = int(pattern.y_min) - 15 if int(pattern.y_min) - 15 > 15 \
                else int(pattern.y_min) + 15
            cv2.putText(frame_out, label, (int(pattern.x_min), ytext), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            if (pattern.is_reliable()):
                x_variance, y_variance = pattern.location_variance()
                x_std2 = 2.0 * sqrt(x_variance)
                y_std2 = 2.0 * sqrt(y_variance)
                cv2.ellipse(frame_out, (int(x_center), int(y_center)), 
                            (int(x_std2),int(y_std2)), 0.0, 0, 360, 
                            pattern.bounding_box_color, 1)
                x_center_velocity, y_center_velocity = pattern.velocity()
                cv2.arrowedLine(frame_out, (int(x_center), int(y_center)), 
                                (int(x_center+x_center_velocity), 
                                 int(y_center+y_center_velocity)), 
                                 pattern.bounding_box_color, 1)


        return True, frame_out

#------------------------------------------------------------------------------
class Detection(BoundingBox):
    """
    Output from object detector, measurement
    """
    def __init__(self, current_time, class_id, x_min, x_max, y_min, y_max, 
                 confidence, image_width, image_height, patch):
        """
        Initialization
        """
        super().__init__(x_min, x_max, y_min, y_max, image_width, image_height,
             patch, confidence)
        # References
        self.pattern = None
        # Attributes
        self.time = current_time
        self.class_id = class_id
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
    def __init__(self, world, time, priority, object_id, text):
        """
        Initialization
        Priority: 0 highest
        """
        # References
        self.world = world
        # Attributes
        self.time = time
        self.priority = priority
        self.object_id = object_id
        self.text = text

#------------------------------------------------------------------------------
class Forecast:
    """
    Forecast of body movement
    """
    def __init__(self, time, body):
        """
        Initialization
        """
        # References
        self.body = body
        # Attributes
        self.time = time
        # Calculate forecast
        min_distance = np.inf
        self.mu = np.zeros((6, FORECAST_COUNT))
        self.sigma = np.zeros((6, 6, FORECAST_COUNT))
        self.mu[0,0] = body.x
        self.mu[1,0] = body.y
        self.mu[2,0] = body.z
        self.mu[3,0] = body.vx
        self.mu[4,0] = body.vy
        self.mu[5,0] = body.vz
        self.sigma[:,:,0] = body.sigma.copy()
        for i in range(1, FORECAST_COUNT):
            self.mu[:,i] = FORECAST_A.dot(self.mu[:,i-1])
            self.sigma[:,:,i] = FORECAST_A.dot(self.sigma[:,:,i-1]).dot(FORECAST_A.T) + BODY_R
            distance = np.linalg.norm(np.array([self.mu[0,i], self.mu[1,i], self.mu[2,i]]))
            if distance < min_distance:
                self.x_min_distance = self.mu[0,i]
                self.y_min_distance = self.mu[1,i]
                self.z_min_distance = self.mu[2,i]
                self.sigma_min_distance = self.sigma[:,:,i].copy()
                self.t_min_distance = time + i * FORECAST_DELTA
                min_distance = distance

#------------------------------------------------------------------------------
class ObjectDetector:
    """
    Deep network object detector for extracting class, bounding box and
    confidence level
    """
    def __init__(self, world, model_file, label_file):
        """
        Initialization
        """
        # References
        self.world = world
        # Attributes
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')        
            
#        self.detection_graph.as_default()
        self.session = tf.Session(graph=self.detection_graph)
        
    def close(self):
        """
        Release resources
        """
        self.session.close()

    def detect(self, image):
        """
        Run inference for single image and return dictionary
        """
        with self.detection_graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            # Run inference
            output_dict = self.session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

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
                         detection.y_max, camera.sensor_width_pixels, 
                         camera.sensor_height_pixels, detection.patch,
                         detection.confidence)
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

        self.x = detection.x
        self.y = detection.y
        self.width = detection.width
        self.height = detection.height
        self.hue0 = detection.hue0
        self.hue1 = detection.hue1
        self.hue2 = detection.hue2
        self.saturation = detection.saturation
        self.value = detection.value
        self.patch = detection.patch

        self.retention_count = 0
        self.matched = False

    def add_detection(self, detection):
        """
        Add detection to list of detections
        """
        self.detections.append(detection)

    def correct(self, detection, delta):
        """
        Correct bounding box coordinates based on new detection and the state
        delta seconds ago. Also updates other attribute information.
        """
        self.confidence = detection.confidence
        #
        k_x_min = self.sigma_x_min.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_x_min).dot(PATTERN_C.T) + PATTERN_Q))
        mu_xmin = np.array([[self.x_min], [self.vx_min]])
        mu_xmin = mu_xmin + k_x_min.dot(detection.x_min - PATTERN_C.dot(mu_xmin))
        self.x_min = mu_xmin[0, 0]
        self.vx_min = mu_xmin[1, 0]
        self.sigma_x_min = (np.eye(2)-k_x_min.dot(PATTERN_C)).dot(self.sigma_x_min)
        #
        k_x_max = self.sigma_x_max.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_x_max).dot(PATTERN_C.T) + PATTERN_Q))
        mu_xmax = np.array([[self.x_max], [self.vx_max]])
        mu_xmax = mu_xmax + k_x_max.dot(detection.x_max - PATTERN_C.dot(mu_xmax))
        self.x_max = mu_xmax[0, 0]
        self.vx_max = mu_xmax[1, 0]
        self.sigma_x_max = (np.eye(2)-k_x_max.dot(PATTERN_C)).dot(self.sigma_x_max)
        #
        k_y_min = self.sigma_y_min.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_y_min).dot(PATTERN_C.T) + PATTERN_Q))
        mu_ymin = np.array([[self.y_min], [self.vy_min]])
        mu_ymin = mu_ymin + k_y_min.dot(detection.y_min - PATTERN_C.dot(mu_ymin))
        self.y_min = mu_ymin[0, 0]
        self.vy_min = mu_ymin[1, 0]
        self.sigma_y_min = (np.eye(2)-k_y_min.dot(PATTERN_C)).dot(self.sigma_y_min)
        #
        k_y_max = self.sigma_y_max.dot(PATTERN_C.T).dot(np.linalg.inv(PATTERN_C.dot(self.sigma_y_max).dot(PATTERN_C.T) + PATTERN_Q))
        mu_ymax = np.array([[self.y_max], [self.vy_max]])
        mu_ymax = mu_ymax + k_y_max.dot(detection.y_max - PATTERN_C.dot(mu_ymax))
        self.y_max = mu_ymax[0, 0]
        self.vy_max = mu_ymax[1, 0]
        self.sigma_y_max = (np.eye(2)-k_y_max.dot(PATTERN_C)).dot(self.sigma_y_max)
        #
        self.set_border_behaviour(self.camera.image_width_pixels, self.camera.image_height_pixels)
        self.matched = True

    def detection(self):
        """
        Get latest detection
        """
        if len(self.detections) == 0:
            return 0.0, 0.0, 0.0, 0.0

        detection = self.detections[-1]
        return detection.x_min, detection.x_max, detection.y_min, detection.y_max
    
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
        self.set_border_behaviour(self.camera.image_width_pixels, 
                                  self.camera.image_height_pixels)
        self.matched = False

    def velocity(self):
        """
        Calculate center point velocity
        """
        x_center = (self.vx_min + self.vx_max) / 2.0
        y_center = (self.vy_min + self.vy_max) / 2.0
        return x_center, y_center

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
class PresentationForecast(Presentation):
    """
    2D map presentation for movement and collision forecast
    """
    def __init__(self, world, map_id, x_loc, y_loc, height_pixels, 
                 width_pixels, extent):
        """
        Initialization
        """
        super().__init__(world)
        self.map_id =map_id
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.height_pixels = height_pixels
        self.width_pixels = width_pixels
        self.extent = extent
        self.frame = np.zeros((height_pixels, width_pixels, 3), np.uint8)
        self.window_name = "Map " + str(map_id)
        cv2.imshow(self.window_name, self.frame)
        cv2.moveWindow(self.window_name, self.x_loc, self.y_loc)
        
    def close(self):
        """
        Release resources
        """
        cv2.destroyWindow(self.window_name)

    def update(self, current_time):
        """
        Update presentation at time t
        """
        extension_pixels = min(self.height_pixels, self.width_pixels)
        pixels_meter = 0.5*extension_pixels / self.extent

        # Empty frame
        self.frame = np.zeros((self.height_pixels, self.width_pixels, 3), 
                              np.uint8)

        # No bodies, now update
        if len(self.world.bodies) == 0:
            return

        # Observer
        color = (255, 255, 255)
        radius_pixels = int((1.63 + sqrt(0.234) )* pixels_meter/2.0)
        cv2.circle(self.frame, (int(self.width_pixels/2), 
                                int(self.height_pixels/2)), 
                                radius_pixels, color, 1)

        # Draw
        for body in self.world.bodies:
            xc = self.width_pixels/2.0 + body.x * pixels_meter
            yc = self.height_pixels/2.0 + body.z * pixels_meter
            color = body.forecast_color
            radius_pixels = int(body.mean_radius() * pixels_meter)
            cv2.circle(self.frame, (int(xc), int(yc)), radius_pixels, color, 1)
            if body.forecast is not None:
                for i in range(1, FORECAST_COUNT):
                    x1 = int(self.width_pixels/2.0 + body.forecast.mu[0,i-1] * pixels_meter)
                    y1 = int(self.height_pixels/2.0 + body.forecast.mu[2,i-1] * pixels_meter)
                    x2 = int(self.width_pixels/2.0 + body.forecast.mu[0,i] * pixels_meter)
                    y2 = int(self.height_pixels/2.0 + body.forecast.mu[2,i] * pixels_meter)
                    cv2.line(self.frame, (x1,y1), (x2,y2), color, 1)
                xc = self.width_pixels/2.0 + body.forecast.x_min_distance * pixels_meter
                yc = self.height_pixels/2.0 + body.forecast.z_min_distance * pixels_meter
                cv2.circle(self.frame, (int(xc), int(yc)), 3, color, 1)
                if body.collision_probability > 0.0:
                    sx = int(sqrt(body.forecast.sigma_min_distance[0,0]))
                    sy = int(sqrt(body.forecast.sigma_min_distance[2,2]))
                    cv2.circle(self.frame, (int(xc), int(yc)), 3, color, 1)
                    cv2.ellipse(self.frame, (int(xc), int(yc)), (sx, sy), 0.0, 0.0, 
                            360.0, color,1)
        cv2.imshow(self.window_name, self.frame)
                
#------------------------------------------------------------------------------
class PresentationLog(Presentation):
    """
    Text file of object history for analyzing data in external software
    Categories:
        Detection
        Pattern
        Body
        Event
    """
    def __init__(self, world, category, text_file):
        """
        Initialization
        """
        super().__init__(world)
        self.category = category
        self.file = open(text_file, "w")
        if self.category == "Detection":
            self.file.write("time,id,class_id,confidence,")
            self.file.write("x_min,x_max,y_min,y_max,")
            self.file.write("border_left,border_right,")
            self.file.write("border_top,border_bottom,")
            self.file.write("pattern,matched")
            self.file.write("\n")
            f = "{0:.3f},{1:d},{2:d},{3:.3f}," # time,id,class_id,confidence
            f += "{4:.3f},{5:.3f},{6:.3f},{7:.3f}," # x_min,x_max,y_min,y_max
            f += "{8:d},{9:d}," # border_left,border_right
            f += "{10:d},{11:d}," # border_top,border_bottom
            f += "{12:d},{13:s}" # pattern,matched
            f += "\n"
            self.fmt = f
        elif self.category == "Pattern":
            self.file.write("time,id,class_id,confidence,")
            self.file.write("x_min,x_max,y_min,y_max,")
            self.file.write("border_left,border_right,")
            self.file.write("border_top,border_bottom,")
            self.file.write("vx_min,vx_max,vy_min,vy_max,")
            self.file.write("sigma_x_min_00,sigma_x_min_01,")
            self.file.write("sigma_x_min_10,sigma_x_min_11,")
            self.file.write("sigma_x_max_00,sigma_x_max_01,")
            self.file.write("sigma_x_max_10,sigma_x_max_11,")
            self.file.write("sigma_y_min_00,sigma_y_min_01,")
            self.file.write("sigma_y_min_10,sigma_y_min_11,")
            self.file.write("sigma_y_max_00,sigma_y_max_01,")
            self.file.write("sigma_y_max_10,sigma_y_max_11,")
            self.file.write("x_center,y_center,")
            self.file.write("vx_center,vy_center,")
            self.file.write("vx_center_var,vy_center_var,")
            self.file.write("retention_count,body,matched,")
            self.file.write("x_min_d,x_max_d,y_min_d,y_max_d")
            self.file.write("\n")
            f = "{0:.3f},{1:d},{2:d},{3:.3f}," # time,id,class_id,confidence
            f += "{4:.3f},{5:.3f},{6:.3f},{7:.3f}," # x_min,x_max,y_min,y_max
            f += "{8:d},{9:d}," # border_left,border_right
            f += "{10:d},{11:d}," # border_top,border_bottom
            f += "{12:.3f},{13:.3f},{14:.3f},{15:.3f}," # vx_min,vx_max,vy_min,vy_max
            f += "{16:.3f},{17:.3f}," # sigma_x_min_00,sigma_x_min_01
            f += "{18:.3f},{19:.3f}," # sigma_x_min_10,sigma_x_min_11
            f += "{20:.3f},{21:.3f}," # sigma_x_max_00,sigma_x_max_01
            f += "{22:.3f},{23:.3f}," # sigma_x_max_10,sigma_x_max_11
            f += "{24:.3f},{25:.3f}," # sigma_y_min_00,sigma_y_min_01
            f += "{26:.3f},{27:.3f}," # sigma_y_min_10,sigma_y_min_11
            f += "{28:.3f},{29:.3f}," # sigma_y_max_00,sigma_y_max_01
            f += "{30:.3f},{31:.3f}," # sigma_y_max_10,sigma_y_max_11
            f += "{32:.3f},{33:.3f}," # x_center,y_center
            f += "{34:.3f},{35:.3f}," # vx_center,vy_center
            f += "{36:.3f},{37:.3f}," # vx_center_var,vy_center_var
            f += "{38:d},{39:d},{40:s}," # retention_count,body,matched
            f += "{41:.3f},{42:.3f},{43:.3f},{44:.3f}" # x_min_d,x_max_d,y_min_d,y_max_d
            f += "\n"
            self.fmt = f
        elif self.category == "Body":
            self.file.write("time,id,class_id,")
            self.file.write("x,y,z,")
            self.file.write("vx,vy,vz,")
            self.file.write("sigma_00,sigma_01,sigma_02,")
            self.file.write("sigma_03,sigma_04,sigma_05,")
            self.file.write("sigma_10,sigma_11,sigma_12,")
            self.file.write("sigma_13,sigma_14,sigma_15,")
            self.file.write("sigma_20,sigma_21,sigma_22,")
            self.file.write("sigma_23,sigma_24,sigma_25,")
            self.file.write("sigma_30,sigma_31,sigma_32,")
            self.file.write("sigma_33,sigma_34,sigma_35,")
            self.file.write("sigma_40,sigma_41,sigma_42,")
            self.file.write("sigma_43,sigma_44,sigma_45,")
            self.file.write("sigma_50,sigma_51,sigma_52,")
            self.file.write("sigma_53,sigma_54,sigma_55,")
            self.file.write("x_pattern,y_pattern,z_pattern,")
            self.file.write("pattern,status,collision_p,")
            self.file.write("c_time,c_x,c_y,c_z,")
            self.file.write("c_sigma_00,c_sigma_01,c_sigma_02,")
            self.file.write("c_sigma_03,c_sigma_04,c_sigma_05,")
            self.file.write("c_sigma_10,c_sigma_11,c_sigma_12,")
            self.file.write("c_sigma_13,c_sigma_14,c_sigma_15,")
            self.file.write("c_sigma_20,c_sigma_21,c_sigma_22,")
            self.file.write("c_sigma_23,c_sigma_24,c_sigma_25,")
            self.file.write("c_sigma_30,c_sigma_31,c_sigma_32,")
            self.file.write("c_sigma_33,c_sigma_34,c_sigma_35,")
            self.file.write("c_sigma_40,c_sigma_41,c_sigma_42,")
            self.file.write("c_sigma_43,c_sigma_44,c_sigma_45,")
            self.file.write("c_sigma_50,c_sigma_51,c_sigma_52,")
            self.file.write("c_sigma_53,c_sigma_54,c_sigma_55")
            self.file.write("\n")
            f = "{0:.3f},{1:d},{2:d}," # time,id,class_id
            f += "{3:.3f},{4:.3f},{5:.3f}," # x,y,z
            f += "{6:.3f},{7:.3f},{8:.3f}," # vx,vy,vz
            f += "{9:.3f},{10:.3f},{11:.3f}," # sigma_00,sigma_01,sigma_02
            f += "{12:.3f},{13:.3f},{14:.3f}," # sigma_03,sigma_04,sigma_05
            f += "{15:.3f},{16:.3f},{17:.3f}," # sigma_10,sigma_11,sigma_12
            f += "{18:.3f},{19:.3f},{20:.3f}," # sigma_13,sigma_14,sigma_15
            f += "{21:.3f},{22:.3f},{23:.3f}," # sigma_20,sigma_21,sigma_22
            f += "{24:.3f},{25:.3f},{26:.3f}," # sigma_23,sigma_24,sigma_25
            f += "{27:.3f},{28:.3f},{29:.3f}," # sigma_30,sigma_31,sigma_42
            f += "{30:.3f},{31:.3f},{32:.3f}," # sigma_33,sigma_34,sigma_45
            f += "{33:.3f},{34:.3f},{35:.3f}," # sigma_40,sigma_41,sigma_42
            f += "{36:.3f},{37:.3f},{38:.3f}," # sigma_43,sigma_44,sigma_45
            f += "{39:.3f},{40:.3f},{41:.3f}," # sigma_50,sigma_51,sigma_52
            f += "{42:.3f},{43:.3f},{44:.3f}," # sigma_53,sigma_54,sigma_55
            f += "{45:.3f},{46:.3f},{47:.3f}," # x_pattern,y_pattern,z_pattern
            f += "{48:d},{49:d},{50:.3f}," # pattern, status, collision_p
            f += "{51:.3f},{52:.3f},{53:.3f},{54:.3f}," # collision time,x,y,z
            f += "{55:.3f},{56:.3f},{57:.3f}," # collision sigma_00,sigma_01,sigma_02
            f += "{58:.3f},{59:.3f},{60:.3f}," # collision sigma_03,sigma_04,sigma_05
            f += "{61:.3f},{62:.3f},{63:.3f}," # collision sigma_10,sigma_11,sigma_12
            f += "{64:.3f},{65:.3f},{66:.3f}," # collision sigma_13,sigma_14,sigma_15
            f += "{67:.3f},{68:.3f},{69:.3f}," # collision sigma_20,sigma_21,sigma_22
            f += "{70:.3f},{71:.3f},{72:.3f}," # collision sigma_23,sigma_24,sigma_25
            f += "{73:.3f},{74:.3f},{75:.3f}," # collision sigma_30,sigma_31,sigma_42
            f += "{76:.3f},{77:.3f},{78:.3f}," # collision sigma_33,sigma_34,sigma_45
            f += "{79:.3f},{80:.3f},{81:.3f}," # collision sigma_40,sigma_41,sigma_42
            f += "{82:.3f},{83:.3f},{84:.3f}," # collision sigma_43,sigma_44,sigma_45
            f += "{85:.3f},{86:.3f},{87:.3f}," # collision sigma_50,sigma_51,sigma_52
            f += "{88:.3f},{89:.3f},{90:.3f}" # collision sigma_53,sigma_54,sigma_55
            f += "\n"
            self.fmt = f
        elif self.category == "Event":
            self.file.write("time,priority,object,text")
            self.file.write("\n")
            f = "{0:.3f},{1:d},{2:d},{3:s}" # time,priority,object,text
            f += "\n"
            self.fmt = f
        else:
            self.file.write("Unsupported category.")
        
    def close(self):
        """
        Release resources
        """
        if self.category == "Event":
            for event in self.world.events:
                self.file.write(self.fmt.format(event.time,
                                                event.priority,
                                                event.object_id,
                                                event.text))
        self.file.close()

    def update(self, current_time):
        """
        Update presentation at time t
        """
        if self.category == "Detection":
            for camera in self.world.cameras:
                for detection in camera.detections:
                    pattern_id = 0
                    if detection.pattern is not None:
                        pattern_id = id(detection.pattern)
                    self.file.write(self.fmt.format(current_time,
                                                    id(detection),
                                                    detection.class_id,
                                                    detection.confidence,
                                                    detection.x_min,
                                                    detection.x_max,
                                                    detection.y_min,
                                                    detection.y_max,
                                                    detection.border_left,
                                                    detection.border_right,
                                                    detection.border_top,
                                                    detection.border_bottom,
                                                    pattern_id,
                                                    str(detection.matched)))
        elif self.category == "Pattern":
            for camera in self.world.cameras:
                for pattern in camera.patterns:                    
                    body_id = 0
                    if pattern.body is not None:
                        body_id = id(pattern.body)
                    x_center,y_center = pattern.center_point()
                    vx_center,vy_center = pattern.velocity()
                    vx_center_var,vy_center_var = pattern.location_variance()
                    x_min_d, x_max_d, y_min_d, y_max_d = pattern.detection()
                    self.file.write(self.fmt.format(current_time,
                                                    id(pattern),
                                                    pattern.class_id,
                                                    pattern.confidence,
                                                    pattern.x_min,
                                                    pattern.x_max,
                                                    pattern.y_min,
                                                    pattern.y_max,
                                                    pattern.border_left,
                                                    pattern.border_right,
                                                    pattern.border_top,
                                                    pattern.border_bottom,
                                                    pattern.vx_min,
                                                    pattern.vx_max,
                                                    pattern.vy_min,
                                                    pattern.vy_max,
                                                    pattern.sigma_x_min[0,0],
                                                    pattern.sigma_x_min[0,1],
                                                    pattern.sigma_x_min[1,0],
                                                    pattern.sigma_x_min[1,1],
                                                    pattern.sigma_x_max[0,0],
                                                    pattern.sigma_x_max[0,1],
                                                    pattern.sigma_x_max[1,0],
                                                    pattern.sigma_x_max[1,1],
                                                    pattern.sigma_y_min[0,0],
                                                    pattern.sigma_y_min[0,1],
                                                    pattern.sigma_y_min[1,0],
                                                    pattern.sigma_y_min[1,1],
                                                    pattern.sigma_y_max[0,0],
                                                    pattern.sigma_y_max[0,1],
                                                    pattern.sigma_y_max[1,0],
                                                    pattern.sigma_y_max[1,1],
                                                    x_center,
                                                    y_center,
                                                    vx_center,
                                                    vy_center,
                                                    vx_center_var,
                                                    vy_center_var,
                                                    pattern.retention_count,
                                                    body_id,
                                                    str(pattern.matched),
                                                    x_min_d,
                                                    x_max_d,
                                                    y_min_d,
                                                    y_max_d))
        elif self.category == "Body":
            for body in self.world.bodies:
                pattern_id = 0
                x_pattern,y_pattern,z_pattern = 0.0, 0.0, 0.0
                if body.pattern is not None:
                    pattern_id = id(body.pattern)
                    x_pattern,y_pattern,z_pattern = body.coordinates_from_pattern()
                status = 0
                if body.status == 'measured':
                    status = 1
                if body.forecast is None:
                    c_time = 0.0
                    c_x = 0.0
                    c_y = 0.0
                    c_z = 0.0
                    s_s_00 = 0.0
                    s_s_01 = 0.0
                    s_s_02 = 0.0
                    s_s_03 = 0.0
                    s_s_04 = 0.0
                    s_s_05 = 0.0
                    s_s_10 = 0.0
                    s_s_11 = 0.0
                    s_s_12 = 0.0
                    s_s_13 = 0.0
                    s_s_14 = 0.0
                    s_s_15 = 0.0
                    s_s_20 = 0.0
                    s_s_21 = 0.0
                    s_s_22 = 0.0
                    s_s_23 = 0.0
                    s_s_24 = 0.0
                    s_s_25 = 0.0
                    s_s_30 = 0.0
                    s_s_31 = 0.0
                    s_s_32 = 0.0
                    s_s_33 = 0.0
                    s_s_34 = 0.0
                    s_s_35 = 0.0
                    s_s_40 = 0.0
                    s_s_41 = 0.0
                    s_s_42 = 0.0
                    s_s_43 = 0.0
                    s_s_44 = 0.0
                    s_s_45 = 0.0
                    s_s_50 = 0.0
                    s_s_51 = 0.0
                    s_s_52 = 0.0
                    s_s_53 = 0.0
                    s_s_54 = 0.0
                    s_s_55 = 0.0
                else:
                    c_time = body.forecast.t_min_distance
                    c_x = body.forecast.x_min_distance
                    c_y = body.forecast.y_min_distance
                    c_z = body.forecast.z_min_distance
                    s_s_00 = body.forecast.sigma_min_distance[0,0]
                    s_s_01 = body.forecast.sigma_min_distance[0,1]
                    s_s_02 = body.forecast.sigma_min_distance[0,2]
                    s_s_03 = body.forecast.sigma_min_distance[0,3]
                    s_s_04 = body.forecast.sigma_min_distance[0,4]
                    s_s_05 = body.forecast.sigma_min_distance[0,5]
                    s_s_10 = body.forecast.sigma_min_distance[1,0]
                    s_s_11 = body.forecast.sigma_min_distance[1,1]
                    s_s_12 = body.forecast.sigma_min_distance[1,2]
                    s_s_13 = body.forecast.sigma_min_distance[1,3]
                    s_s_14 = body.forecast.sigma_min_distance[1,4]
                    s_s_15 = body.forecast.sigma_min_distance[1,5]
                    s_s_20 = body.forecast.sigma_min_distance[2,0]
                    s_s_21 = body.forecast.sigma_min_distance[2,1]
                    s_s_22 = body.forecast.sigma_min_distance[2,2]
                    s_s_23 = body.forecast.sigma_min_distance[2,3]
                    s_s_24 = body.forecast.sigma_min_distance[2,4]
                    s_s_25 = body.forecast.sigma_min_distance[2,5]
                    s_s_30 = body.forecast.sigma_min_distance[3,0]
                    s_s_31 = body.forecast.sigma_min_distance[3,1]
                    s_s_32 = body.forecast.sigma_min_distance[3,2]
                    s_s_33 = body.forecast.sigma_min_distance[3,3]
                    s_s_34 = body.forecast.sigma_min_distance[3,4]
                    s_s_35 = body.forecast.sigma_min_distance[3,5]
                    s_s_40 = body.forecast.sigma_min_distance[4,0]
                    s_s_41 = body.forecast.sigma_min_distance[4,1]
                    s_s_42 = body.forecast.sigma_min_distance[4,2]
                    s_s_43 = body.forecast.sigma_min_distance[4,3]
                    s_s_44 = body.forecast.sigma_min_distance[4,4]
                    s_s_45 = body.forecast.sigma_min_distance[4,5]
                    s_s_50 = body.forecast.sigma_min_distance[5,0]
                    s_s_51 = body.forecast.sigma_min_distance[5,1]
                    s_s_52 = body.forecast.sigma_min_distance[5,2]
                    s_s_53 = body.forecast.sigma_min_distance[5,3]
                    s_s_54 = body.forecast.sigma_min_distance[5,4]
                    s_s_55 = body.forecast.sigma_min_distance[5,5]
                self.file.write(self.fmt.format(current_time,
                                                id(body),
                                                body.class_id,
                                                body.x,
                                                body.y,
                                                body.z,
                                                body.vx,
                                                body.vy,
                                                body.vz,
                                                body.sigma[0,0],
                                                body.sigma[0,1],
                                                body.sigma[0,2],
                                                body.sigma[0,3],
                                                body.sigma[0,4],
                                                body.sigma[0,5],
                                                body.sigma[1,0],
                                                body.sigma[1,1],
                                                body.sigma[1,2],
                                                body.sigma[1,3],
                                                body.sigma[1,4],
                                                body.sigma[1,5],
                                                body.sigma[2,0],
                                                body.sigma[2,1],
                                                body.sigma[2,2],
                                                body.sigma[2,3],
                                                body.sigma[2,4],
                                                body.sigma[2,5],
                                                body.sigma[3,0],
                                                body.sigma[3,1],
                                                body.sigma[3,2],
                                                body.sigma[3,3],
                                                body.sigma[3,4],
                                                body.sigma[3,5],
                                                body.sigma[4,0],
                                                body.sigma[4,1],
                                                body.sigma[4,2],
                                                body.sigma[4,3],
                                                body.sigma[4,4],
                                                body.sigma[4,5],
                                                body.sigma[5,0],
                                                body.sigma[5,1],
                                                body.sigma[5,2],
                                                body.sigma[5,3],
                                                body.sigma[5,4],
                                                body.sigma[5,5],
                                                x_pattern,
                                                y_pattern,
                                                z_pattern,
                                                pattern_id,
                                                status,
                                                body.collision_probability,
                                                c_time,
                                                c_x,
                                                c_y,
                                                c_z,
                                                s_s_00,
                                                s_s_01,
                                                s_s_02,
                                                s_s_03,
                                                s_s_04,
                                                s_s_05,
                                                s_s_10,
                                                s_s_11,
                                                s_s_12,
                                                s_s_13,
                                                s_s_14,
                                                s_s_15,
                                                s_s_20,
                                                s_s_21,
                                                s_s_22,
                                                s_s_23,
                                                s_s_24,
                                                s_s_25,
                                                s_s_30,
                                                s_s_31,
                                                s_s_32,
                                                s_s_33,
                                                s_s_34,
                                                s_s_35,
                                                s_s_40,
                                                s_s_41,
                                                s_s_42,
                                                s_s_43,
                                                s_s_44,
                                                s_s_45,
                                                s_s_50,
                                                s_s_51,
                                                s_s_52,
                                                s_s_53,
                                                s_s_54,
                                                s_s_55))
        else:
            pass
                
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
        # Attributes

    def say(self, text):
        """
        Say the text
        """
#        self.engine.say(text)
#        self.engine.runAndWait()
        
    def alarm(self):
        """
        Make alarm sound
        """
        winsound.Beep(1500, 200)
#------------------------------------------------------------------------------
class Trace:
    """
    History of body movement and other changes
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
        self.camera_left = None
        self.camera_right = None
        self.presentations = []
        self.events = []
        self.speech_synthesizer = SpeechSynthesizer(self)
        self.object_detector = ObjectDetector(self, PATH_TO_CKPT, '')
        # Attributes
        self.current_time = 0.0
        self.current_frame = 0
        self.delta_time = 1.0/10.0
        self.last_forecast = -10.0
        self.mode = "step"

        self.frame = np.zeros((UI_Y4+1, UI_X3+1, 3), np.uint8)
        label = "Commands:"
        cv2.putText(self.frame, label, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1)
        label = "   s: step one frame"
        cv2.putText(self.frame, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1)
        label = "   c: continuous mode"
        cv2.putText(self.frame, label, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1)
        label = "   q: quit"
        cv2.putText(self.frame, label, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 255, 255), 1)

        cv2.imshow("ShadowWorld", self.frame)
        cv2.moveWindow("ShadowWorld", 10, 10)

        self.check_keyboard_command()
        if self.mode == "quit":
            self.close()

    def add_body(self, body):
        """
        Add a body
        """
        self.bodies.append(body)
        self.add_event(Event(self, self.current_time, 1, id(body),
                             "Body created"))

    def add_camera_left(self, camera):
        """
        Adds a camera
        """
        self.camera_left = camera
        
    def add_camera_right(self, camera):
        """
        Adds a camera
        """
        self.camera_right = camera

    def add_event(self, event):
        """
        Add an event
        """
        self.events.append(event)
        if event.priority < 0:
            self.speech_synthesizer.alarm()
        if event.priority <= 0:
            self.speech_synthesizer.say(event.text)

    def add_presentation(self, presentation):
        """
        Adds a presentation
        """
        self.presentations.append(presentation)
        
    def check_keyboard_command(self):
        """
        Wait for keyboard command and set the required mode
        """
        if self.mode == "step":
            found = False
            while not found: # Discard everything except n, q, c
                key_pushed = cv2.waitKey(0) & 0xFF
                if key_pushed in [ord('s'), ord('q'), ord('c')]:
                    found = True
            if key_pushed == ord('q'):
                self.mode = "quit"
            if key_pushed == ord('s'):
                self.mode = "step"
            if key_pushed == ord('c'):
                self.mode = "continuous"
        else:
            key_pushed = cv2.waitKey(1) & 0xFF
            if key_pushed == ord('q'):
                self.mode = "quit"
            if key_pushed == ord('s'):
                self.mode = "step"

    def check_patterns_for_bodies(self):
        """
        Match camera patterns and create bodies for (new) matches
        """
        patterns_left = self.camera_left.patterns
        n_left = len(patterns_left)
        l_features = np.zeros((n_left, 10))
        for i in range(0, n_left):
            pat = patterns_left[i]
            l_features[i, 0] = pat.confidence
            l_features[i, 1] = pat.x
            l_features[i, 2] = pat.y
            l_features[i, 3] = pat.width
            l_features[i, 4] = pat.height
            l_features[i, 5] = pat.hue0
            l_features[i, 6] = pat.hue1
            l_features[i, 7] = pat.hue2
            l_features[i, 8] = pat.saturation
            l_features[i, 9] = pat.value
            
        patterns_right = self.camera_right.patterns
        n_right = len(patterns_right)
        r_features = np.zeros((n_right, 10))
        for i in range(0, n_right):
            pat = patterns_right[i]
            r_features[i, 0] = pat.confidence
            r_features[i, 1] = pat.x
            r_features[i, 2] = pat.y
            r_features[i, 3] = pat.width
            r_features[i, 4] = pat.height
            r_features[i, 5] = pat.hue0
            r_features[i, 6] = pat.hue1
            r_features[i, 7] = pat.hue2
            r_features[i, 8] = pat.saturation
            r_features[i, 9] = pat.value
        
        cost = np.zeros((n_left, n_right))
        for row in range(0, n_left):
            for col in range(0, n_right):
                p = multivariate_normal.pdf(l_features[row]-r_features[col],
                                            mean=STEREO_MATCH_MEAN, 
                                            cov=STEREO_MATCH_COVARIANCE)
                p = -np.log(p)
                if np.isinf(p):
                    p = 999.0
                cost[row][col] = p

        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            lpat = patterns_left[row_ind[i]]
            rpat = patterns_right[col_ind[i]]
            if lpat.body is None and rpat.body is None:
                if cost[row_ind[i]][col_ind[i]]>STEREO_MIN_LOG_PROBABILITY:
                    new_body = Body(self, lpat, rpat)
                    lpat.body = new_body
                    rpat.body = new_body
                    self.add_body(new_body)

    def close(self):
        """
        Release resources
        """
        self.object_detector.close()
        for camera in [self.camera_left, self.camera_right]:
            camera.close()
        for presentation in self.presentations:
            presentation.close()
        cv2.destroyWindow("ShadowWorld")

    def run(self):
        """
        Run the world
        """
#        self.check_keyboard_command()
#        if self.mode == "quit":
#            return
        # Empty frame
        self.frame = np.zeros((UI_Y4+1, UI_X3+1, 3), np.uint8)
        
        more = True
        while more:
            more = True
            frame_offset = 0
            # Camera views
            for camera in [self.camera_left, self.camera_right]:
                more_camera, camera_frame = camera.update(self.current_time, 
                                                          self.delta_time)
                if not more_camera:
                    return
                else:
                    camera_frame = cv2.resize(camera_frame, (0,0), 
                                              fx=UI_VIDEO_SCALE, 
                                              fy=UI_VIDEO_SCALE)
                    cxmin = 0
                    cxmax = UI_X1
                    cymin = frame_offset
                    cymax = frame_offset + (UI_Y1-UI_Y0)
                    self.frame[cymin:cymax, cxmin:cxmax, :] = camera_frame
                    frame_offset += (UI_Y1-UI_Y0)

            # Stereo view
            camera1 = self.camera_left
            camera2 = self.camera_right
            stereo = cv2.StereoBM_create(numDisparities=1*16)
            disp_rgb = stereo.compute(
                cv2.cvtColor(camera1.current_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(camera2.current_frame, cv2.COLOR_BGR2GRAY))
            minValue = disp_rgb.min()
            maxValue = disp_rgb.max()
            disp_rgb = np.uint8(255 * (disp_rgb - minValue) / (maxValue - minValue))
            disp = cv2.applyColorMap(disp_rgb, cv2.COLORMAP_WINTER)
            cxmin = int(UI_VIDEO_SCALE*(camera1.sensor_width_pixels-camera1.image_width_pixels)/2)
            cxmax = UI_X1 -cxmin
            cymin = UI_Y2+int(UI_VIDEO_SCALE*(camera1.sensor_height_pixels-camera1.image_height_pixels)/2)
            cymax = UI_Y4 - (cymin-UI_Y2)
            camera_frame = cv2.resize(disp, (cxmax-cxmin,cymax-cymin))
            self.frame[cymin:cymax, cxmin:cxmax, :] = camera_frame

            self.check_patterns_for_bodies()
            
            for body in self.bodies:
                body.predict(self.delta_time)
                    
                if body.frame_age == BODY_DATA_COLLECTION_COUNT:
                    text = CLASS_NAMES[body.class_id]+" observed "
                    speed = int(3.6*body.speed())
                    distance = int(abs(body.z))
                    text += "distance " + str(distance) + " meters "
                    if (speed < 4):
                        text += "hanging around "
                    else:
                        if body.z <= 0 and body.vz >= 0:
                            text += "moving towards us "
                        elif body.z <= 0 and body.vz < 0:
                            text += "moving away from us "
                        if body.z > 0 and body.vz >= 0:
                            text += "moving away from us "
                        elif body.z > 0 and body.vz < 0:
                            text += "moving towards us "
                        if body.vx <= 0:
                            text += "from right to left "
                        else:
                            text += "from left to right "
                        text += "speed " + str(speed) + " kilometers per hour "

                    self.add_event(Event(self, self.current_time, 0, id(body),
                                         text))
                if body.pattern_left is not None:
                    if body.pattern_left.is_reliable() and body.pattern_right.is_reliable():
                        xo, yo, zo = body.coordinates_from_patterns()
                        body.correct(xo, yo, zo, self.delta_time)

            for body in self.bodies:
                body.add_trace(self.current_time)

            if (self.current_time > self.last_forecast + FORECAST_INTERVAL):
                for body in self.bodies:
                    body.make_forecast(self.current_time)
                    body.detect_collision()
                    if body.collision_probability > COLLISION_MIN_LEVEL:
                        if body.collision_probability > body.collision_probability_max:
                            t_collision = int(10*(body.forecast.t_min_distance-self.current_time)/10)
                            if t_collision > 0:
                                text = CLASS_NAMES[body.class_id] + " may collide in "
                                text += str(t_collision)
                                text += " seconds with probability "
                                text += str(int(1000*body.collision_probability)/10)
                                text += " percent"
                                self.add_event(Event(self, self.current_time, -1, id(body), text))
                            body.collision_probability_max = body.collision_probability
                       
                self.last_forecast = self.current_time


            """
            Update map (from above)
            """
            # Empty frame
            map_frame = np.zeros((UI_Y3-UI_Y0, UI_X2-UI_X1, 3), np.uint8)
            xc = int((UI_X2-UI_X1)/2)
            yc = int((UI_Y3-UI_Y0)/2)
            pixels_meter = xc / MAP_EXTENT
            # Radar circles
            radius = 10.0
            while radius < MAP_EXTENT:
                radius_pixels = int(radius * pixels_meter)
                cv2.circle(map_frame, (xc, yc), radius_pixels, (200,200,200), 1)
                radius += 10.0
            # Camera field of views
            for camera in [self.camera_left, self.camera_right]:
                x_offset = int(camera.x * pixels_meter)
                x = xc * tan(camera.field_of_view/2.0)
                pt1 = (int(xc+x_offset), int(yc))
                pt2 = (int(xc-x+x_offset),0)
                cv2.line(map_frame, pt1, pt2, (200,200,200), 1)
                pt2 = (int(xc+x+x_offset),0)
                cv2.line(map_frame, pt1, pt2, (200,200,200), 1)
            #Bodies
            for body in self.bodies:
                xcc = xc + body.x * pixels_meter
                ycc = yc + body.z * pixels_meter
                color = (0,255,0) # green
                if body.status == "predicted":
                    color = (0,0,255) # red
                radius_pixels = int(body.mean_radius() * pixels_meter)
                cv2.circle(map_frame, (int(xcc), int(ycc)), radius_pixels, color, 1)
                color = (100,100,100)
                sx = int(sqrt(body.sigma[0,0]))
                sy = int(sqrt(body.sigma[2,2]))
                cv2.ellipse(map_frame, (int(xcc), int(ycc)), (sx, sy), 0.0, 0.0, 
                            360.0, color,1)
            self.frame[UI_Y0:UI_Y3, UI_X1:UI_X2, :] = map_frame

            """
            Update map (from side)
            """
            # Empty frame
            map_frame = np.zeros((UI_Y4-UI_Y3, UI_X2-UI_X1, 3), np.uint8)
            xc = int((UI_X2-UI_X1)/2)
            yc = int((UI_Y4-UI_Y3)/2)
            pixels_meter = xc / MAP_EXTENT
            # Radar circles
            radius = 10.0
            while radius < MAP_EXTENT:
                radius_pixels = int(radius * pixels_meter)
                cv2.circle(map_frame, (xc, yc), radius_pixels, (200,200,200), 1)
                radius += 10.0
            # Ground
            cv2.line(map_frame,(0,yc), (2*xc,yc), (200,200,200), 1)
            #Bodies
            for body in self.bodies:
                xcc = xc + body.x * pixels_meter
                ycc = yc - body.y * pixels_meter
                color = (0,255,0) # green
                if body.status == "predicted":
                    color = (0,0,255) # red
                radius_pixels = int(body.mean_radius() * pixels_meter)
                cv2.circle(map_frame, (int(xcc), int(ycc)), radius_pixels, color, 1)
                color = (100,100,100)
                sx = int(sqrt(body.sigma[0,0]))
                sy = int(sqrt(body.sigma[2,2]))
                cv2.ellipse(map_frame, (int(xcc), int(ycc)), (sx, sy), 0.0, 0.0, 
                            360.0, color,1)
            self.frame[UI_Y3:UI_Y4, UI_X1:UI_X2, :] = map_frame

            """
            Report
            """
            # Empty frame
            rep_frame = np.zeros((UI_Y4-UI_Y0, UI_X3-UI_X2, 3), np.uint8)
            irow = 15
            offset = 15
            # Draw heading
            label = "Time {0:<.2f}, frame {1:d}".format(self.current_time, self.current_frame)
            cv2.putText(rep_frame, label, (10,irow), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            irow += offset
            #Bodies
            label = "Bodies:"
            cv2.putText(rep_frame, label, (10,irow), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            irow += offset
            for body in self.bodies:
                label = "{0:6s} {1:5.1f} {2:5.1f} {3:5.1f} {4:5.1f} {5:5.1f} {6:5.1f}".format(
                        CLASS_NAMES[body.class_id], 
                        body.x, body.y, body.z,
                        body.vx, body.vy, body.vz)
                cv2.putText(rep_frame, label, (10,irow), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                irow += offset
            
            self.frame[UI_Y0:UI_Y4, UI_X2:UI_X3, :] = rep_frame
            

            line_color = 60
            # Horizontal lines
            self.frame[UI_Y0, UI_X0:UI_X3, :] = line_color
            self.frame[UI_Y1, UI_X0:UI_X1, :] = line_color
            self.frame[UI_Y2, UI_X0:UI_X1, :] = line_color
            self.frame[UI_Y3, UI_X1:UI_X2, :] = line_color
            self.frame[UI_Y4, UI_X0:UI_X3, :] = line_color
            # Vertical lines
            self.frame[UI_Y0:UI_Y4, UI_X0, :] = line_color
            self.frame[UI_Y0:UI_Y4, UI_X1, :] = line_color
            self.frame[UI_Y0:UI_Y4, UI_X2, :] = line_color
            self.frame[UI_Y0:UI_Y4, UI_X3, :] = line_color
            cv2.imshow("ShadowWorld", self.frame)

            self.check_keyboard_command()
            if self.mode == "quit":
                return
                break

#------------------------------------------------------------------------------
def run_application():
    """
    Example application
    """

    # Load KITTI data
    basedir = 'D:\Thesis\Kitti\Raw'
    date = '2011_09_26'
    drive = '0009'
    dataset = pykitti.raw(basedir, date, drive, imformat='cv2')

    world = World()
    
    sensor_width = SENSOR_WIDTH * 4.65 / 1000000 # ICX267
    sensor_height = SENSOR_HEIGHT * 4.65 / 1000000 # ICX267

    camera1 = Camera(world, name="Left", focal_length=0.003,
                     sensor_width=sensor_width, sensor_height=sensor_height, 
                     sensor_width_pixels=SENSOR_WIDTH, 
                     sensor_height_pixels=SENSOR_HEIGHT,
                     x=0.0, y=0.0, z=0.0, 
                     yaw=0.0, pitch=0.0, roll=0.0, 
                     frames=dataset.cam2)
#                     x_loc=20, y_loc=20, width_pixels=900, 
#                     height_pixels=int(900*720/1280))
    world.add_camera_left(camera1)

    camera2 = Camera(world, name="Right", focal_length=0.003,
                     sensor_width=sensor_width, sensor_height=sensor_height, 
                     sensor_width_pixels=SENSOR_WIDTH, 
                     sensor_height_pixels=SENSOR_HEIGHT,
                     x=0.54, y=0.0, z=0.0, 
                     yaw=0.0, pitch=0.0, roll=0.0, 
                     frames=dataset.cam3)
#                     x_loc=20, y_loc=20, width_pixels=900, 
#                     height_pixels=int(900*720/1280))
    world.add_camera_right(camera2)
#
#    world.add_presentation(PresentationForecast(world, map_id=2, x_loc=940, 
#                                                y_loc=620, height_pixels=500, 
#                                                width_pixels=500, 
#                                                extent=TEST_EXTENTS[test_video]))

    world.add_presentation(PresentationLog(world, "Detection", "logs/DetectionStereo.txt"))
    world.add_presentation(PresentationLog(world, "Pattern", "logs/PatternStereo.txt"))
    world.add_presentation(PresentationLog(world, "Body", "logs/BodyStereo.txt"))
    world.add_presentation(PresentationLog(world, "Event", "logs/EventStereo.txt"))

    world.run()
    
    world.close()

if __name__ == "__main__":
    run_application()
    