# -*- coding: utf-8 -*-
"""
ImageObject class definitions

All attribute units are in SI (m, m/s, m/s**2).
'Height', 'witdth' and 'length' are defined as looking the subject directly
from the front. 'Length' corresponds to 'depth'.
Created on Wed Nov 29 09:08:16 2017
@author: Sakari Lampola
"""

from math import inf, nan, atan, cos, sqrt
import random as rnd
import numpy as np
import SpeechSynthesis as ss
from scipy.optimize import linear_sum_assignment

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

SIMILARITY_DISTANCE = 0.4 # Max distance to size ratio for similarity interpretation
CONFIDENFE_LEVEL_CREATE = 0.80 # How confident we must be to create a new object
CONFIDENFE_LEVEL_UPDATE = 0.20 # How confident we must be to update existing object
#BORDER_WIDTH_CREATE = 30 # Image objects are not allowed to born if near image border
#BORDER_WIDTH_REMOVE = 10 # Image objects are removed if near image border
BORDER_WIDTH = 30 # Part of screen for special image object behaviour
RETENTION_COUNT_MAX = 30 # How many frames and image object is kept untedected

def next_id(category):
    """
    Generate a unique id
    """
    if (category is 'detected_object'):
        next_id.detected_object_counter += 1
        return next_id.detected_object_counter
    elif (category is 'image_object'):
        next_id.image_object_counter += 1
        return next_id.image_object_counter
    else:
        return 0

next_id.detected_object_counter = 0
next_id.image_object_counter = 0

def check_bounding_box_location(x_min, x_max, y_min, y_max, image_width, image_height, tolerance):
    """
    Returns true if the box is in allowed region
    """
    c1 = x_min < tolerance
    c2 = x_max > (image_width - tolerance)
    c3 = y_min < tolerance
    c4 = y_max > (image_height - tolerance)
    return not(c1 or c2 or c3 or c4)

class DetectedObject:
    """
    Raw object detection information
    """
    def __init__(self, time, class_type, x_min, x_max, y_min, y_max, confidence, \
                 appearance):
        """
        Initialization
        """
        self.id = next_id('detected_object')
        self.time = time
        self.class_type = class_type
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.appearance = appearance
        self.confidence = confidence
        self.matched = False

    def distance(self, image_object):
        """
        Calculates distance (4 corners) to a predicted image object
        """
        dx_min = abs(self.x_min - image_object.x_min)
        dx_max = abs(self.x_max - image_object.x_max)
        dy_min = abs(self.y_min - image_object.y_min)
        dy_max = abs(self.y_max - image_object.y_max)
        
        horizontal_size = max(image_object.x_max - image_object.x_min, 1)
        vertical_size = max(image_object.y_max - image_object.y_min, 1)
        
        dx_min /= horizontal_size
        dx_max /= horizontal_size
        dy_min /= vertical_size
        dy_max /= vertical_size
        
        d = (dx_min + dx_max + dy_min + dy_max) / 4.0 # Average of 4 corners

        return d 

    def distance_with_class(self, image_object):
        """
        Calculates distance to a predicted image object including class
        """
        d = self.distance(image_object)

        # Change of class penalized
        if self.class_type != image_object.class_type:
            d += 100.0

        return d 


IMAGE_OBJECT_R1 = 0.1 # Image object state equation location variance
IMAGE_OBJECT_R2 = 1.0 # Image object state equation velocity variance
IMAGE_OBJECT_R = np.array([[IMAGE_OBJECT_R1, 0.0],
                           [0.0, IMAGE_OBJECT_R2]]) # Image object state equation covariance matrix

IMAGE_OBJECT_Q1 = 200.0 # Image object (location) measurement variance
IMAGE_OBJECT_Q = np.array([IMAGE_OBJECT_Q1])

IMAGE_OBJECT_ALFA = 200.0 # Image object initial location error variance
IMAGE_OBJECT_BETA = 10000.0 # Image object initial velocity error variance

IMAGE_OBJECT_C = np.array([[1.0, 0.0]]) # Image object measurement matrix

class ImageObject:
    """
    Generic base class for different image objects
    """
    name = 'generic'
    class_type = 0
    height_min, height_mean, height_max = 0.0, nan, inf
    width_min, width_mean, width_max = 0.0, nan, inf
    length_min, length_mean, length_max = 0.0, nan, inf
    velocity_max, acceleration_max = inf, inf

    def __init__(self, detected_object, image_world):
        """
        Initialization
        """
        self.image_world = image_world
        self.color = (rnd.randint(0,255),rnd.randint(0,255),rnd.randint(0,255)) # ui color
        self.matched = False # detected object matching

        self.id = next_id('image_object')
        self.detected = True
        self.border_left = 1 # Inside margin area, normal behaviour
        self.border_right = 1 # Inside margin area, normal behaviour
        self.border_top = 1 # Inside margin area, normal behaviour
        self.border_bottom = 1 # Inside margin area, normal behaviour

        self.x_min = detected_object.x_min
        self.x_max = detected_object.x_max
        self.y_min = detected_object.y_min
        self.y_max = detected_object.y_max

        self.set_border_behaviour()

        self.vx_min = 0.0
        self.vx_max = 0.0
        self.vy_min = 0.0
        self.vy_max = 0.0
        
        self.sigma_x_min = np.array([[IMAGE_OBJECT_ALFA, 0],[0.0, IMAGE_OBJECT_BETA]])
        self.sigma_x_max = np.array([[IMAGE_OBJECT_ALFA, 0],[0.0, IMAGE_OBJECT_BETA]])
        self.sigma_y_min = np.array([[IMAGE_OBJECT_ALFA, 0],[0.0, IMAGE_OBJECT_BETA]])
        self.sigma_y_max = np.array([[IMAGE_OBJECT_ALFA, 0],[0.0, IMAGE_OBJECT_BETA]])
        
        self.confidence = detected_object.confidence
        self.appearance = detected_object.appearance
        self.retention_count = 0
        
        self.x_camera = 0.0 
        self.y_camera = 0.0 
        self.z_camera = 0.0 

    def center_point(self):
        """
        Calculate the bounding box center point
        """
        x_center = (self.x_min + self.x_max) / 2.0
        y_center = (self.y_min + self.y_max) / 2.0
        
        return x_center, y_center

    def center_point_velocity(self):
        """
        Calculate the bounding box center point
        """
        x_center = (self.vx_min + self.vx_max) / 2.0
        y_center = (self.vy_min + self.vy_max) / 2.0
        
        return x_center, y_center

    def location_variance(self):
        """
        Calculate the bounding box center point
        """
        x_variance = self.sigma_x_min[0,0] + self.sigma_x_max[0,0]
        y_variance = self.sigma_y_min[0,0] + self.sigma_y_max[0,0]
        
        return x_variance, y_variance
    
    def is_center_reliable(self):
        """
        Decides whethet center point is reliable
        """
        if self.border_left in (3,5,6):
            return False

        if self.border_right in (3,5,6):
            return False

        if self.border_top in (3,5,6):
            return False

        if self.border_bottom in (3,5,6):
            return False

        return True

    def border_extent(self):
        """
        Counts the number of borders the object touches
        """
        extent = 0

        if self.border_left > 1:
            extent  += 1
        if self.border_right > 1:
            extent  += 1
        if self.border_top > 1:
            extent  += 1
        if self.border_bottom > 1:
            extent  += 1
        
        return extent

    def is_vanished(self):
        """
        Checks if the image object size goes zero or negative
        """
        c1 = (self.x_max <= self.x_min)
        c2 = (self.y_max <= self.y_min)
        if c1 or c2:
            return True
        return False
    
    def set_border_behaviour(self):
        """
        Sets the border according to:
            1 = normal area
            2 = normal + border area
            3 = normal + out of screen area
            4 = inside border area
            5 = border + out of screen area
            6 = out of screen area
        """
        image_width = self.image_world.width
        image_height = self.image_world.height
        # left
        self.border_left = 1
        if (self.x_max >= BORDER_WIDTH and \
            self.x_min <= BORDER_WIDTH and \
            self.x_min >= 0):
            self.border_left = 2
        if (self.x_max >= BORDER_WIDTH and \
            self.x_min <= 0):
            self.border_left = 3
        if (self.x_max <= BORDER_WIDTH and \
            self.x_min <= BORDER_WIDTH and \
            self.x_min >= 0):
            self.border_left = 4
        if (self.x_max <= BORDER_WIDTH and \
            self.x_min <= 0):
            self.border_left = 5
        if (self.x_min <= 0 and \
            self.x_max <= 0):
            self.border_left = 6
        # right
        self.border_right = 1
        if (self.x_min <= (image_width - BORDER_WIDTH) and \
            self.x_max >= (image_width - BORDER_WIDTH) and \
            self.x_max <= image_width):
            self.border_right = 2
        if (self.x_min <= (image_width - BORDER_WIDTH) and \
            self.x_max >= image_width):
            self.border_right = 3
        if (self.x_min >= (image_width - BORDER_WIDTH) and \
            self.x_max >= (image_width - BORDER_WIDTH) and \
            self.x_max <= image_width):
            self.border_right = 4
        if (self.x_min >= (image_width - BORDER_WIDTH) and \
            self.x_max >= image_width):
            self.border_right = 5
        if (self.x_min >= image_width and \
            self.x_max >= image_width):
            self.border_right = 6
        # top
        self.border_top = 1
        if (self.y_max >= BORDER_WIDTH and \
            self.y_min <= BORDER_WIDTH and \
            self.y_min >= 0):
            self.border_top = 2
        if (self.y_max >= BORDER_WIDTH and \
            self.y_min <= 0):
            self.border_top = 3
        if (self.y_max <= BORDER_WIDTH and \
            self.y_min <= BORDER_WIDTH and \
            self.y_min >= 0):
            self.border_top = 4
        if (self.y_max <= BORDER_WIDTH and \
            self.y_min <= 0):
            self.border_top = 5
        if (self.y_min <= 0 and \
            self.y_max <= 0):
            self.border_top = 6
        # bottom
        self.border_bottom = 1
        if (self.y_min <= (image_height - BORDER_WIDTH) and \
            self.y_max >= (image_height - BORDER_WIDTH) and \
            self.y_max <= image_height):
            self.border_bottom = 2
        if (self.y_min <= (image_height - BORDER_WIDTH) and \
            self.y_max >= image_height):
            self.border_bottom = 3
        if (self.y_min >= (image_height - BORDER_WIDTH) and \
            self.y_max >= (image_height - BORDER_WIDTH) and \
            self.y_max <= image_height):
            self.border_bottom = 4
        if (self.y_min >= (image_height - BORDER_WIDTH) and \
            self.y_max >= image_height):
            self.border_bottom = 5
        if (self.y_min >= image_height and \
            self.y_max >= image_height):
            self.border_bottom = 6
    
    def predict(self, delta):
        """
        Predict bounding box coordinates based on Kalman filtering
        """
        a = np.array([[1.0, delta],[0.0, 1.0]]) # state equation matrix

        self.set_border_behaviour()

#        if (self.border_left not in [3,5]):
#            mu_xmin = a.dot(np.array([[self.x_min],[self.vx_min]]))
#        else:
#            mu_xmin = a.dot(np.array([[self.x_min],[self.vx_max]]))
        mu_xmin = a.dot(np.array([[self.x_min],[self.vx_min]]))
        self.x_min = mu_xmin[0,0]
        self.vx_min = mu_xmin[1,0]
        self.sigma_x_min = a.dot(self.sigma_x_min).dot(a.T) + IMAGE_OBJECT_R

#        if (self.border_right not in [3,5]):
#            mu_xmax = a.dot(np.array([[self.x_max],[self.vx_max]]))
#        else:
#            mu_xmax = a.dot(np.array([[self.x_max],[self.vx_min]]))
        mu_xmax = a.dot(np.array([[self.x_max],[self.vx_max]]))
        self.x_max = mu_xmax[0,0]
        self.vx_max = mu_xmax[1,0]
        self.sigma_x_max = a.dot(self.sigma_x_max).dot(a.T) + IMAGE_OBJECT_R

#        if (self.border_top not in [3,5]):
#            mu_ymin = a.dot(np.array([[self.y_min],[self.vy_min]]))
#        else:
#            mu_ymin = a.dot(np.array([[self.y_min],[self.vy_max]]))
        mu_ymin = a.dot(np.array([[self.y_min],[self.vy_min]]))
        self.y_min = mu_ymin[0,0]
        self.vy_min = mu_ymin[1,0]
        self.sigma_y_min = a.dot(self.sigma_y_min).dot(a.T) + IMAGE_OBJECT_R

#        if (self.border_bottom not in [3,5]):
#            mu_ymax = a.dot(np.array([[self.y_max],[self.vy_max]]))
#        else:
#            mu_ymax = a.dot(np.array([[self.y_max],[self.vy_min]]))
        mu_ymax = a.dot(np.array([[self.y_max],[self.vy_max]]))
        self.y_max = mu_ymax[0,0]
        self.vy_max = mu_ymax[1,0]
        self.sigma_y_max = a.dot(self.sigma_y_max).dot(a.T) + IMAGE_OBJECT_R

        self.set_border_behaviour()

        self.matched = False # By default, not matched with any detected object
        
    def correct(self, detected_object, delta):
        """
        Correct bounding box coordinates based on detected object measurement
        """
        self.confidence = detected_object.confidence
        self.set_border_behaviour()

        a = np.array([[1.0, delta],[0.0, 1.0]]) # state equation matrix

#        if (self.border_left not in [3,5]):
        k_x_min = self.sigma_x_min.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_x_min).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_xmin = a.dot(np.array([[self.x_min],[self.vx_min]]))
        mu_xmin = mu_xmin + k_x_min.dot(detected_object.x_min - IMAGE_OBJECT_C.dot(mu_xmin))
        self.x_min = mu_xmin[0,0]
        self.vx_min = mu_xmin[1,0]
        self.sigma_x_min = (np.eye(2)-k_x_min.dot(IMAGE_OBJECT_C)).dot(self.sigma_x_min)

#        if (self.border_right not in [3,5]):
        k_x_max = self.sigma_x_max.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_x_max).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_xmax = a.dot(np.array([[self.x_max],[self.vx_max]]))
        mu_xmax = mu_xmax + k_x_max.dot(detected_object.x_max - IMAGE_OBJECT_C.dot(mu_xmax))
        self.x_max = mu_xmax[0,0]
        self.vx_max = mu_xmax[1,0]
        self.sigma_x_max = (np.eye(2)-k_x_max.dot(IMAGE_OBJECT_C)).dot(self.sigma_x_max)

#        if (self.border_top not in [3,5]):
        k_y_min = self.sigma_y_min.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_y_min).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_ymin = a.dot(np.array([[self.y_min],[self.vy_min]]))
        mu_ymin = mu_ymin + k_y_min.dot(detected_object.y_min - IMAGE_OBJECT_C.dot(mu_ymin))
        self.y_min = mu_ymin[0,0]
        self.vy_min = mu_ymin[1,0]
        self.sigma_y_min = (np.eye(2)-k_y_min.dot(IMAGE_OBJECT_C)).dot(self.sigma_y_min)

#        if (self.border_bottom not in [3,5]):
        k_y_max = self.sigma_y_max.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_y_max).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_ymax = a.dot(np.array([[self.y_max],[self.vy_max]]))
        mu_ymax = mu_ymax + k_y_max.dot(detected_object.y_max - IMAGE_OBJECT_C.dot(mu_ymax))
        self.y_max = mu_ymax[0,0]
        self.vy_max = mu_ymax[1,0]
        self.sigma_y_max = (np.eye(2)-k_y_max.dot(IMAGE_OBJECT_C)).dot(self.sigma_y_max)

        self.set_border_behaviour()

        self.matched = True

class Aeroplane(ImageObject):
    """
    Aeroplane - derived class
    """
    name = 'aeroplane'
    class_type = 1
    height_min, height_mean, height_max = 1.5, 2.0, 6.0
    width_min, width_mean, width_max = 2.0, 20.0, 80.0
    length_min, length_mean, length_max = 3.0, 20.0, 80.0
    velocity_max, acceleration_max = 300.0, 10.0

class Bicycle(ImageObject):
    """
    Bicycle - derived class
    """
    name = 'bicycle'
    class_type = 2
    height_min, height_mean, height_max = 0.6, 1.1, 1.4
    width_min, width_mean, width_max = 0.4, 0.6, 0.8
    length_min, length_mean, length_max = 1.2, 1.5, 1.7
    velocity_max, acceleration_max = 13.0, 1.0

class Bird(ImageObject):
    """
    Bird - derived class
    """
    name = 'bird'
    class_type = 3
    height_min, height_mean, height_max = 0.1, 0.25, 2.8
    width_min, width_mean, width_max = 0.05, 0.5, 1.2
    length_min, length_mean, length_max = 0.1, 0.2, 1.2
    velocity_max, acceleration_max = 80.0, 100.0

class Boat(ImageObject):
    """
    Boat - derived class
    """
    name = 'boat'
    class_type = 4
    height_min, height_mean, height_max = 1.0, 2.0, 20.0
    width_min, width_mean, width_max = 0.6, 3.0, 60.0
    length_min, length_mean, length_max = 1.5, 3.0, 350.0
    velocity_max, acceleration_max = 20.0, 5.0

class Bottle(ImageObject):
    """
    Bottle - derived class
    """
    name = 'bottle'
    class_type = 5
    height_min, height_mean, height_max = 0.1, 0.25, 0.5
    width_min, width_mean, width_max = 0.05, 0.1, 0.3
    length_min, length_mean, length_max = 0.05, 0.1, 0.3
    velocity_max, acceleration_max = 10.0, 3.0

class Bus(ImageObject):
    """
    Bus - derived class
    """
    name = 'bus'
    class_type = 6
    height_min, height_mean, height_max = 2.5, 2.8, 5.0
    width_min, width_mean, width_max = 2.3, 2.5, 2.8
    length_min, length_mean, length_max = 6.0, 15.0, 30.0
    velocity_max, acceleration_max = 30.0, 2.0

class Car(ImageObject):
    """
    Car - derived class
    """
    name = 'car'
    class_type = 7
    height_min, height_mean, height_max = 1.3, 1.5, 2.0
    width_min, width_mean, width_max = 1.6, 1.8, 2.0
    length_min, length_mean, length_max = 3.5, 4.3, 5.5
    velocity_max, acceleration_max = 50.0, 10.0

class Cat(ImageObject):
    """
    Cat - derived class
    """
    name = 'cat'
    class_type = 8
    height_min, height_mean, height_max = 0.2, 0.3, 0.4
    width_min, width_mean, width_max = 0.1, 0.15, 0.2
    length_min, length_mean, length_max = 0.3, 0.4, 0.5
    velocity_max, acceleration_max = 13.0, 10.0

class Chair(ImageObject):
    """
    Chair - derived class
    """
    name = 'chair'
    class_type = 9
    height_min, height_mean, height_max = 0.5, 1.0, 1.5
    width_min, width_mean, width_max = 0.5, 0.6, 0.7
    length_min, length_mean, length_max = 0.5, 0.6, 0.7
    velocity_max, acceleration_max = 10.0, 3.0

class Cow(ImageObject):
    """
    Cow - derived class
    """
    name = 'cow'
    class_type = 10
    height_min, height_mean, height_max = 1.3, 1.3, 1.5
    width_min, width_mean, width_max = 0.5, 0.6, 0.7
    length_min, length_mean, length_max = 1.8, 2.3, 2.8
    velocity_max, acceleration_max = 10.0, 2.0

class DiningTable(ImageObject):
    """
    Dining table - derived class
    """
    name = 'dining table'
    class_type = 11
    height_min, height_mean, height_max = 0.7, 0.75, 0.9
    width_min, width_mean, width_max = 0.5, 1.0, 1.5
    length_min, length_mean, length_max = 0.5, 2.0, 5.0
    velocity_max, acceleration_max = 10.0, 3.0

class Dog(ImageObject):
    """
    Dog - derived class
    """
    name = 'dog'
    class_type = 12
    height_min, height_mean, height_max = 0.2, 0.35, 0.8
    width_min, width_mean, width_max = 0.15, 0.2, 0.3
    length_min, length_mean, length_max = 0.3, 0.6, 1.3
    velocity_max, acceleration_max = 13.0, 10.0

class Horse(ImageObject):
    """
    Horse - derived class
    """
    name = 'horse'
    class_type = 13
    height_min, height_mean, height_max = 1.2, 1.6, 1.8
    width_min, width_mean, width_max = 0.3, 0.45, 0.6
    length_min, length_mean, length_max = 2.0, 2.4, 3.0
    velocity_max, acceleration_max = 13.0, 4.0

class Motorbike(ImageObject):
    """
    Motorbike - derived class
    """
    name = 'motorbike'
    class_type = 14
    height_min, height_mean, height_max = 0.6, 1.1, 1.4
    width_min, width_mean, width_max = 0.4, 0.6, 0.8
    length_min, length_mean, length_max = 1.2, 1.5, 1.7
    velocity_max, acceleration_max = 50.0, 10.0

class Person(ImageObject):
    """
    Person - derived class
    """
    name = 'person'
    class_type = 15
    height_min, height_mean, height_max = 1.0, 1.7, 2.0
    width_min, width_mean, width_max = 0.5, 0.6, 0.7
    length_min, length_mean, length_max = 0.25, 0.35, 0.45
    velocity_max, acceleration_max = 10.0, 3.0

class PottedPlant(ImageObject):
    """
    Potted plant - derived class
    """
    name = 'potted plant'
    class_type = 16
    height_min, height_mean, height_max = 0.2, 0.35, 1.0
    width_min, width_mean, width_max = 0.15, 0.25, 0.4
    length_min, length_mean, length_max = 0.15, 0.25, 0.4
    velocity_max, acceleration_max = 10.0, 3.0

class Sheep(ImageObject):
    """
    Sheep - derived class
    """
    name = 'sheep'
    class_type = 17
    height_min, height_mean, height_max = 0.8, 1.2, 1.4
    width_min, width_mean, width_max = 0.25, 0.35, 0.45
    length_min, length_mean, length_max = 1.0, 1.3, 1.5
    velocity_max, acceleration_max = 10.0, 6.0

class Sofa(ImageObject):
    """
    Sofa - derived class
    """
    name = 'sofa'
    class_type = 18
    height_min, height_mean, height_max = 0.7, 1.0, 1.4
    width_min, width_mean, width_max = 0.7, 1.0, 1.5
    length_min, length_mean, length_max = 1.5, 2.0, 4.0
    velocity_max, acceleration_max = 10.0, 3.0

class Train(ImageObject):
    """
    Train - derived class
    """
    name = 'train'
    class_type = 19
    height_min, height_mean, height_max = 2.5, 3.0, 5.0
    width_min, width_mean, width_max = 2.5, 3.1, 3.5
    length_min, length_mean, length_max = 25.0, 150.0, 1000.0
    velocity_max, acceleration_max = 50.0, 1.0

class TVMonitor(ImageObject):
    """
    TV Monitor - derived class
    """
    name = 'tv monitor'
    class_type = 20
    height_min, height_mean, height_max = 0.2, 0.5, 1.2
    width_min, width_mean, width_max = 0.3, 0.8, 2.0
    length_min, length_mean, length_max = 0.04, 0.1, 0.5
    velocity_max, acceleration_max = 10.0, 3.0

class Event:
    """
    Event to be translated into language
    """
    def __init__(self, time, text, priority, world):
        """
        Initialization
        """
        self.time = time
        self.text = text
        self.priority = priority
        self.world = world
        self.world.speech_synthesizer.say(text)

class ImageWorld:
    """
    List of image objects with internal state and ability to predict.
    """
    def __init__(self, width, height):
        """
        Initialization
        """
        self.image_objects = [] # In the beginning, the world is empty...

        self.width = width # image size in pixels
        self.height = height # image height in pixels

        self.focal_length = 0.050 # meters
        self.sensor_width = 0.0359 # meters
        self.sensor_height = 0.0240 # meters
        self.fov = 2.0*atan(self.sensor_width / (2.0*self.focal_length)) # field of view
        
        self.speech_synthesizer = ss.SpeechSynthesizer()
        self.events = [] # In the beginning, no events

    def get_camera_coordinates_from_image_object(self, image_object):
        """
        Calculates camera coordinates from image object center point
        coordinates and camera parameters
        """
        sw = self.sensor_width
        sh = self.sensor_height
        pw = self.width
        ph = self.height
        hi = image_object.y_max - image_object.y_min
        h = image_object.height_mean
        f = self.focal_length
        xp, yp = image_object.center_point()
        
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
        

    def add_event(self, time, text, priority):
        """
        Create a new events to be spelled out
        """
        event = Event(time, text, priority, self)
        self.events.append(event)

    def create_image_object(self, detected_object):
        """
        Create a new object based on detected object information
        """
        if detected_object.confidence < CONFIDENFE_LEVEL_CREATE:
            return False

        new_object = None
        if detected_object.class_type == 1:
            new_object = Aeroplane(detected_object, self)
        elif detected_object.class_type == 2:
            new_object = Bicycle(detected_object, self)
        elif detected_object.class_type == 3:
            new_object = Bird(detected_object, self)
        elif detected_object.class_type == 4:
            new_object = Boat(detected_object, self)
        elif detected_object.class_type == 5:
            new_object = Bottle(detected_object, self)
        elif detected_object.class_type == 6:
            new_object = Bus(detected_object, self)
        elif detected_object.class_type == 7:
            new_object = Car(detected_object, self)
        elif detected_object.class_type == 8:
            new_object = Cat(detected_object, self)
        elif detected_object.class_type == 9:
            new_object = Chair(detected_object, self)
        elif detected_object.class_type == 10:
            new_object = Cow(detected_object, self)
        elif detected_object.class_type == 11:
            new_object = DiningTable(detected_object, self)
        elif detected_object.class_type == 12:
            new_object = Dog(detected_object, self)
        elif detected_object.class_type == 13:
            new_object = Horse(detected_object, self)
        elif detected_object.class_type == 14:
            new_object = Motorbike(detected_object, self)
        elif detected_object.class_type == 15:
            new_object = Person(detected_object, self)
        elif detected_object.class_type == 16:
            new_object = PottedPlant(detected_object, self)
        elif detected_object.class_type == 17:
            new_object = Sheep(detected_object, self)
        elif detected_object.class_type == 18:
            new_object = Sofa(detected_object, self)
        elif detected_object.class_type == 19:
            new_object = Train(detected_object, self)
        elif detected_object.class_type == 20:
            new_object = TVMonitor(detected_object, self)
            
            
        # Don't create if not in predefined class
        if new_object is None:
            return False

        # Don't create if in border area or out of screen
        if new_object.border_left > 3 or new_object.border_right > 3 or \
            new_object.border_top > 3 or new_object.border_bottom > 3:
            return False

        # Don't create if touching 3 or more borders
        if new_object.border_extent() > 2:
            return False
        
        speech = new_object.name
        speech += " " + str(new_object.id)
        conf = int(detected_object.confidence*100.0)
        speech += " confidence " + str(conf)
        speech += " observed"
        self.add_event(detected_object.time, speech, 1)
        self.image_objects.append(new_object)

        return True

    def remove_image_object(self, image_object, time):
        """
        Remove image object from the world

        """
        speech = image_object.name
        speech += " " + str(image_object.id)
        speech += " disappeared"
        self.add_event(time, speech, 1)
        self.image_objects.remove(image_object)

    def update(self, detection_time, detected_objects, log_file, trace_file, time_step):
        """
        Detected objects are taken into consideration. Previously observed
        image objects are matched to the new detections. If no match
        is found, a new image object is created. Matched image objects are
        updated. Image objects not matched are removed.
        """

#        # If the detected object location is in border area, the object will be removed
#        removes = []
#        for detected_object in detected_objects:
#            if not check_bounding_box_location(detected_object.x_min, detected_object.x_max, 
#                                  detected_object.y_min, detected_object.y_max,
#                                  self.width, self.height, BORDER_WIDTH_REMOVE):
#                log_file.write("Delected object {0:d} removed due to being in border area:\n".format(detected_object.id))
#                log_file.write("---{0:6.2f} {1:7.2f} {2:7.2f} {3:7.2f} {4:7.2f}\n".format(
#                    detected_object.confidence, detected_object.x_min,
#                    detected_object.x_max, detected_object.y_min, detected_object.y_max))
#
#                removes.append(detected_object)
#
#        # Delete removed objects
#        for remove in removes:
#            detected_objects.remove(remove)

        # Process previous image objects
        if len(self.image_objects) > 0:
            # Predict new coordinates for each image object
            log_file.write("Image objects ({0:d}), predicted new locations:\n".format(len(self.image_objects)))
            for image_object in self.image_objects:
                image_object.predict(time_step)
                
                log_file.write("---{0:d} {1:s} {2:6.2f} {3:7.2f} {4:7.2f} {5:7.2f} {6:7.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f} {14:.2f}\n".format(
                    image_object.id, image_object.name, image_object.confidence,
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max, \
                    float(image_object.appearance[0]), float(image_object.appearance[1]), \
                    float(image_object.appearance[2]), float(image_object.appearance[3]), \
                    float(image_object.appearance[4]), float(image_object.appearance[5]), \
                    float(image_object.appearance[6]), float(image_object.appearance[7])))
                
            # If the predicted location is out of screen it will be removed
            removes = []
            for image_object in self.image_objects:
                if image_object.border_left == 6 or image_object.border_right == 6 or image_object.border_top == 6 or image_object.border_bottom == 6:
                    log_file.write("Image object {0:d} removed due to being out of screen:\n".format(image_object.id))
                    log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                        image_object.name, image_object.confidence, image_object.x_min,
                        image_object.x_max, image_object.y_min, image_object.y_max))
    
                    removes.append(image_object)
    
            # Delete removed objects
            for remove in removes:
                self.remove_image_object(remove, detection_time)

        # Match detected objects to image objects
        if len(self.image_objects) > 0 and len(detected_objects) > 0:
            # Calculate cost matrix for the Hungarian algorithm (assignment problem)
            log_file.write("Original cost matrix for Hungarian algorithm:\n")
            cost = np.zeros((len(detected_objects), len(self.image_objects)))
            detected_object_index = 0
            for detected_object in detected_objects:
                log_file.write("{0:6d} ".format(detected_object.id))
                image_object_index = 0
                found = False # Just to decide whether to print a newline into the log
                for image_object in self.image_objects:
                    cost[detected_object_index, image_object_index] = \
                        detected_object.distance_with_class(image_object)
                    log_file.write("{0:7.2f} ({1:d}) ".format(cost[detected_object_index, image_object_index], image_object.id))
                    image_object_index += 1
                    found = True
                detected_object_index += 1
                if found:
                    log_file.write("\n")
    
            # Remove rows and columns with no values inside SIMILARITY_DISTANCE
            count_detected_objects = len(detected_objects)        
            count_image_objects = len(self.image_objects)
    
            remove_detected_objects = []
            for detected_object_index in range(0, count_detected_objects):
                found = False
                for image_object_index in range(0, count_image_objects):
                    if cost[detected_object_index, image_object_index] < SIMILARITY_DISTANCE:
                        found = True
                if found == False:
                    remove_detected_objects.append(detected_objects[detected_object_index]) 
            remove_image_objects = []
            for image_object_index in range(0, count_image_objects):
                found = False
                for detected_object_index in range(0, count_detected_objects):
                    if cost[detected_object_index, image_object_index] < SIMILARITY_DISTANCE:
                        found = True
                if found == False:
                    remove_image_objects.append(self.image_objects[image_object_index]) 
    
            detected_objects_to_match = []
            image_objects_to_match = []
    
            for detected_object in detected_objects:
                if detected_object not in remove_detected_objects:
                    detected_objects_to_match.append(detected_object)
                
            for image_object in self.image_objects:
                if image_object not in remove_image_objects:
                    image_objects_to_match.append(image_object)
    
            # The optimal assignment without far-away objects:
            cost_to_match = np.zeros((len(detected_objects_to_match), \
                                      len(image_objects_to_match)))
    
            row_index = -1
            row_index_original = 0
            for detected_object in detected_objects:
                if detected_object not in remove_detected_objects:
                    row_index += 1
                    col_index = -1
                    col_index_original = 0
                    for image_object in self.image_objects:
                        if image_object not in remove_image_objects:
                            col_index +=1
                            cost_to_match[row_index, col_index] = cost[row_index_original, col_index_original]
                        col_index_original += 1
                row_index_original += 1
    
            # Let's print the modified cost matrix
            log_file.write("Modified cost matrix for Hungarian algorithm:\n")
            detected_object_index = 0
            for detected_object in detected_objects_to_match:
                log_file.write("{0:6d} ".format(detected_object.id))
                image_object_index = 0
                found = False # Just to decide whether to print a newline into the log
                for image_object in image_objects_to_match:
                    log_file.write("{0:7.2f} ({1:d}) ".format(cost_to_match[detected_object_index, image_object_index], image_object.id))
                    image_object_index += 1
                    found = True
                detected_object_index += 1
                if found:
                    log_file.write("\n")
           
            # Find the optimal assigment
            row_ind, col_ind = linear_sum_assignment(cost_to_match)
            
            # Update matched objects
            log_file.write("Optimal assignment:\n")
            match_index = 0
            for row in row_ind:
                i1 = row_ind[match_index]
                i2 = col_ind[match_index]
                detected_object = detected_objects_to_match[i1]
                image_object = image_objects_to_match[i2]
                log_file.write("Detected object {0:d} matched to image object {1:d}\n".format(detected_object.id, \
                               image_object.id))
                distance = cost_to_match[i1,i2]
                if distance > SIMILARITY_DISTANCE:
                    log_file.write("Detected object {0:d} - image object {1:d}, distance too far!\n".format(\
                                   detected_object.id, image_object.id))
                    match_index += 1
                else:
                    log_file.write("Existing image object {0:d} updated:\n".format(image_object.id))
        
                    x_min_predicted = image_object.x_min
                    x_max_predicted = image_object.x_max
                    y_min_predicted = image_object.y_min
                    y_max_predicted = image_object.y_max
    
                    vx_min_predicted = image_object.vx_min
                    vx_max_predicted = image_object.vx_max
                    vy_min_predicted = image_object.vy_min
                    vy_max_predicted = image_object.vy_max
        
                    log_file.write("---predicted: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f}\n".format(
                        image_object.name, image_object.confidence, \
                        image_object.x_min, image_object.x_max, \
                        image_object.y_min, image_object.y_max,
                        float(image_object.appearance[0]), float(image_object.appearance[1]), \
                        float(image_object.appearance[2]), float(image_object.appearance[3]), \
                        float(image_object.appearance[4]), float(image_object.appearance[5]), \
                        float(image_object.appearance[6]), float(image_object.appearance[7])))
        
                    log_file.write("---measured:  {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f} {14:.2f}\n".format( \
                        CLASS_NAMES[detected_object.class_type], detected_object.confidence, \
                        float(detected_object.x_min), float(detected_object.x_max), \
                        float(detected_object.y_min), float(detected_object.y_max), \
                        float(detected_object.appearance[0]), float(detected_object.appearance[1]), \
                        float(detected_object.appearance[2]), float(detected_object.appearance[3]), \
                        float(detected_object.appearance[4]), float(detected_object.appearance[5]), \
                        float(detected_object.appearance[6]), float(detected_object.appearance[7]), \
                        distance))
                    
                    image_object.correct(detected_object, time_step)
                    image_object.detected = True
                    image_object.retention_count = 0
                    detected_object.matched = True
        
                    log_file.write("---corrected: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f}\n".format(
                        image_object.name, image_object.confidence, \
                        image_object.x_min, image_object.x_max, \
                        image_object.y_min, image_object.y_max,
                        float(image_object.appearance[0]), float(image_object.appearance[1]), \
                        float(image_object.appearance[2]), float(image_object.appearance[3]), \
                        float(image_object.appearance[4]), float(image_object.appearance[5]), \
                        float(image_object.appearance[6]), float(image_object.appearance[7])))
        
                    fmt = ("{0:.3f},{1:d},{2:.3f},{3:.3f},{4:.3f},{5:.3f},"
                           "{6:.3f},{7:.3f},{8:.3f},{9:.3f},{10:.3f},{11:.3f},"
                           "{12:.3f},{13:.3f},{14:.3f},"
                           "{15:.3f},{15:.3f},{17:.3f},{18:.3f},{19:.3f},{20:.3f},{21:.3f},{22:.3f},"
                           "{23:.2f},{24:.2f},{25:.2f},{26:.2f},{27:.2f},{28:.2f},{29:.2f},{30:.2f},"
                           "{31:.2f},{32:.2f},{33:.2f},{34:.2f},{35:.2f},{36:.2f},{37:.2f},{38:.2f},{39:.2f}\n")
        
                    trace_file.write(fmt.format(detection_time, \
                                                image_object.id, \
                                                x_min_predicted, \
                                                x_max_predicted, \
                                                y_min_predicted, \
                                                y_max_predicted, \
                                                detected_object.x_min, \
                                                detected_object.x_max, \
                                                detected_object.y_min, \
                                                detected_object.y_max, \
                                                image_object.x_min, \
                                                image_object.x_max, \
                                                image_object.y_min, \
                                                image_object.y_max, \
                                                vx_min_predicted, \
                                                vx_max_predicted, \
                                                vy_min_predicted, \
                                                vy_max_predicted, \
                                                image_object.vx_min, \
                                                image_object.vx_max, \
                                                image_object.vy_min, \
                                                image_object.vy_max, \
                                                distance, \
                                                float(detected_object.appearance[0]), \
                                                float(detected_object.appearance[1]), \
                                                float(detected_object.appearance[2]), \
                                                float(detected_object.appearance[3]), \
                                                float(detected_object.appearance[4]), \
                                                float(detected_object.appearance[5]), \
                                                float(detected_object.appearance[6]), \
                                                float(detected_object.appearance[7]), \
                                                float(image_object.appearance[0]), \
                                                float(image_object.appearance[1]), \
                                                float(image_object.appearance[2]), \
                                                float(image_object.appearance[3]), \
                                                float(image_object.appearance[4]), \
                                                float(image_object.appearance[5]), \
                                                float(image_object.appearance[6]), \
                                                float(image_object.appearance[7]), \
                                                image_object.confidence))
                    match_index += 1
                
            # If the corrected location is out of screen it will be removed
            removes = []
            for image_object in self.image_objects:
                if image_object.border_left == 6 or image_object.border_right == 6 or image_object.border_top == 6 or image_object.border_bottom == 6:
                    log_file.write("Image object {0:d} removed due to being pot of screen:\n".format(image_object.id))
                    log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                        image_object.name, image_object.confidence, image_object.x_min,
                        image_object.x_max, image_object.y_min, image_object.y_max))
    
                    removes.append(image_object)
    
            # Delete removed objects
            for remove in removes:
                self.remove_image_object(remove, detection_time)
    
    #        # If the corrected location is in border area, the object will be removed
    #        removes = []
    #        for image_object in self.image_objects:
    #            if not check_bounding_box_location(image_object.x_min, image_object.x_max, 
    #                                  image_object.y_min, image_object.y_max,
    #                                  self.width, self.height, BORDER_WIDTH_REMOVE):
    #                log_file.write("Image object {0:d} removed due to being in border area:\n".format(image_object.id))
    #                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
    #                    image_object.name, image_object.confidence, image_object.x_min,
    #                    image_object.x_max, image_object.y_min, image_object.y_max))
    #
    #                removes.append(image_object)
    #
    #        # Delete removed objects
    #        for remove in removes:
    #            self.image_objects.remove(remove)

        # Remove vanished image objects
        removes = []
        for image_object in self.image_objects:
            if image_object.is_vanished():
                log_file.write("Image object {0:d} removed due to being vanished:\n".format(image_object.id))
                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))

                removes.append(image_object)

        # Delete removed objects
        for remove in removes:
            self.remove_image_object(remove, detection_time)

        # Remove objects touching 3 or more borders
        removes = []
        for image_object in self.image_objects:
            if image_object.border_extent() > 2:
                log_file.write("Image object {0:d} removed due to touching 3 borders:\n".format(image_object.id))
                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))

                removes.append(image_object)

        # Delete removed objects
        for remove in removes:
            self.remove_image_object(remove, detection_time)

        # Any image object not matched is changed to not detected state
        for image_object in self.image_objects:
            if not image_object.matched:
                log_file.write("Image object {0:d} status changed to not detected:\n".format(image_object.id))
                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))

                image_object.detected = False
                image_object.retention_count += 1


        # Remove objects that has not been detected for some time
        removes = []
        for image_object in self.image_objects:
            if image_object.retention_count > RETENTION_COUNT_MAX:
                log_file.write("Image object {0:d} removed due to being not detected for a while:\n".format(image_object.id))
                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))

                removes.append(image_object)

        # Delete removed objects
        for remove in removes:
            self.remove_image_object(remove, detection_time)

#        # Any image object not matched is removed
#        removes = []
#        for image_object in self.image_objects:
#            if not image_object.matched:
#                log_file.write("Image object {0:d} removed:\n".format(image_object.id))
#
#                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
#                    image_object.name, image_object.confidence, image_object.x_min,
#                    image_object.x_max, image_object.y_min, image_object.y_max))
#
#                removes.append(image_object)
#
#        # Delete removed objects
#        for remove in removes:
#            self.image_objects.remove(remove)

        # If no match for a detected object, create a new image object
        for detected_object in detected_objects:
            if not detected_object.matched:
                found = False
                for image_object in self.image_objects:
                    if detected_object.distance(image_object) < SIMILARITY_DISTANCE:
                        found = True
                if not found: # only if there is no other image object near
                    created = self.create_image_object(detected_object)
                    if created: # if inside border area and we are confident enough
                        log_file.write("Image object {0:d} created: ".format(next_id.image_object_counter))
                        log_file.write("{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                            CLASS_NAMES[detected_object.class_type], detected_object.confidence,
                            detected_object.x_min, detected_object.x_max, \
                            detected_object.y_min, detected_object.y_max))


        # Update image object camera coordinates
        for image_object in self.image_objects:
            image_object.x_camera, image_object.y_camera, image_object.z_camera = \
                self.get_camera_coordinates_from_image_object(image_object)
        