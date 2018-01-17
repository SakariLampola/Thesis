# -*- coding: utf-8 -*-
"""
ImageObject class definitions

All attribute units are in SI (m, m/s, m/s**2).
'Height', 'witdth' and 'length' are defined as looking the subject directly
from the front. 'Length' corresponds to 'depth'.
Created on Wed Nov 29 09:08:16 2017
@author: Sakari Lampola
"""

from math import inf, nan
import random as rnd
import numpy as np
from scipy.optimize import linear_sum_assignment

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

SIMILARITY_DISTANCE = 1.0 # Max distance to size ratio for similarity interpretation
RETENTION_TIME = 0.0 # How long image objects are maintained without new detections
CONFIDENFE_LEVEL = 0.0 # How confident we must be to create a new object

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

#def kalman_filter_predict(mu, sigma, a, r):
#    mu_new = a.dot(mu)
#    sigma_new = a.dot(sigma.dot(a.T))+r
#    return mu_new, sigma_new
#
#def kalman_filter_correct(mu, sigma, c, q, z):
#    k = sigma.dot(c.T).dot(np.linalg.inv(c.dot(sigma).dot(c.T)+q))
#    mu_new = mu + k.dot(z - c.dot(mu))
#    sigma_new = (np.eye(12)-k.dot(c)).dot(sigma)
#    return mu_new, sigma_new

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

        return (dx_min + dx_max + dy_min + dy_max) / 4.0 # Average of 4 corners

IMAGE_OBJECT_R1 = 1.0 # Image object state equation location variance
IMAGE_OBJECT_R2 = 1.0 # Image object state equation velocity variance
IMAGE_OBJECT_R = np.array([[IMAGE_OBJECT_R1, 0.0],
                           [0.0, IMAGE_OBJECT_R2]]) # Image object state equation covariance matrix

IMAGE_OBJECT_Q1 = 10.0 # Image object (location) measurement variance
IMAGE_OBJECT_Q = np.array([IMAGE_OBJECT_Q1])

IMAGE_OBJECT_ALFA = 10.0 # Image object initial location error variance
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
        self.status = 'visible'

        self.x_min = detected_object.x_min
        self.x_max = detected_object.x_max
        self.y_min = detected_object.y_min
        self.y_max = detected_object.y_max

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
    
    def predict(self, delta):
        """
        Predict bounding box coordinates based on Kalman filtering
        """
        a = np.array([[1.0, delta],[0.0, 1.0]]) # state equation matrix

        mu_xmin = a.dot(np.array([[self.x_min],[self.vx_min]]))
        self.x_min = mu_xmin[0,0]
        self.vx_min = mu_xmin[1,0]
        self.sigma_x_min = a.dot(self.sigma_x_min).dot(a.T) + IMAGE_OBJECT_R

        mu_xmax = a.dot(np.array([[self.x_max],[self.vx_max]]))
        self.x_max = mu_xmax[0,0]
        self.vx_max = mu_xmax[1,0]
        self.sigma_x_max = a.dot(self.sigma_x_max).dot(a.T) + IMAGE_OBJECT_R

        mu_ymin = a.dot(np.array([[self.y_min],[self.vy_min]]))
        self.y_min = mu_ymin[0,0]
        self.vy_min = mu_ymin[1,0]
        self.sigma_y_min = a.dot(self.sigma_y_min).dot(a.T) + IMAGE_OBJECT_R

        mu_ymax = a.dot(np.array([[self.y_max],[self.vy_max]]))
        self.y_max = mu_ymax[0,0]
        self.vy_max = mu_ymax[1,0]
        self.sigma_y_max = a.dot(self.sigma_y_max).dot(a.T) + IMAGE_OBJECT_R

        self.matched = False # By default, not matched with any detected object
        
    def correct(self, detected_object, delta):
        """
        Correct bounding box coordinates based on detected object measurement
        """
        self.confidence = detected_object.confidence

        a = np.array([[1.0, delta],[0.0, 1.0]]) # state equation matrix

        k_x_min = self.sigma_x_min.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_x_min).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_xmin = a.dot(np.array([[self.x_min],[self.vx_min]]))
        mu_xmin = mu_xmin + k_x_min.dot(detected_object.x_min - IMAGE_OBJECT_C.dot(mu_xmin))
        self.x_min = mu_xmin[0,0]
        self.vx_min = mu_xmin[1,0]
        self.sigma_x_min = (np.eye(2)-k_x_min.dot(IMAGE_OBJECT_C)).dot(self.sigma_x_min)

        k_x_max = self.sigma_x_max.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_x_max).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_xmax = a.dot(np.array([[self.x_max],[self.vx_max]]))
        mu_xmax = mu_xmax + k_x_max.dot(detected_object.x_max - IMAGE_OBJECT_C.dot(mu_xmax))
        self.x_max = mu_xmax[0,0]
        self.vx_max = mu_xmax[1,0]
        self.sigma_x_max = (np.eye(2)-k_x_max.dot(IMAGE_OBJECT_C)).dot(self.sigma_x_max)

        k_y_min = self.sigma_y_min.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_y_min).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_ymin = a.dot(np.array([[self.y_min],[self.vy_min]]))
        mu_ymin = mu_ymin + k_y_min.dot(detected_object.y_min - IMAGE_OBJECT_C.dot(mu_ymin))
        self.y_min = mu_ymin[0,0]
        self.vy_min = mu_ymin[1,0]
        self.sigma_y_min = (np.eye(2)-k_y_min.dot(IMAGE_OBJECT_C)).dot(self.sigma_y_min)

        k_y_max = self.sigma_y_max.dot(IMAGE_OBJECT_C.T).\
            dot(np.linalg.inv(IMAGE_OBJECT_C.dot(self.sigma_y_max).dot(IMAGE_OBJECT_C.T) + IMAGE_OBJECT_Q))
        mu_ymax = a.dot(np.array([[self.y_max],[self.vy_max]]))
        mu_ymax = mu_ymax + k_y_max.dot(detected_object.y_max - IMAGE_OBJECT_C.dot(mu_ymax))
        self.y_max = mu_ymax[0,0]
        self.vy_max = mu_ymax[1,0]
        self.sigma_y_max = (np.eye(2)-k_y_max.dot(IMAGE_OBJECT_C)).dot(self.sigma_y_max)

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

class ImageWorld:
    """
    List of image objects with internal state and ability to predict.
    """
    def __init__(self, width, height):
        """
        Initialization
        """
        self.image_objects = [] # In the beginning, the world is empty...
        self.width = width
        self.height =height


    def create_image_object(self, detected_object):
        """
        Create a new object based on detected object information
        """
        if detected_object.class_type == 1:
            self.image_objects.append(Aeroplane(detected_object, self))
        elif detected_object.class_type == 2:
            self.image_objects.append(Bicycle(detected_object, self))
        elif detected_object.class_type == 3:
            self.image_objects.append(Bird(detected_object, self))
        elif detected_object.class_type == 4:
            self.image_objects.append(Boat(detected_object, self))
        elif detected_object.class_type == 5:
            self.image_objects.append(Bottle(detected_object, self))
        elif detected_object.class_type == 6:
            self.image_objects.append(Bus(detected_object, self))
        elif detected_object.class_type == 7:
            self.image_objects.append(Car(detected_object, self))
        elif detected_object.class_type == 8:
            self.image_objects.append(Cat(detected_object, self))
        elif detected_object.class_type == 9:
            self.image_objects.append(Chair(detected_object, self))
        elif detected_object.class_type == 10:
            self.image_objects.append(Cow(detected_object, self))
        elif detected_object.class_type == 11:
            self.image_objects.append(DiningTable(detected_object, self))
        elif detected_object.class_type == 12:
            self.image_objects.append(Dog(detected_object, self))
        elif detected_object.class_type == 13:
            self.image_objects.append(Horse(detected_object, self))
        elif detected_object.class_type == 14:
            self.image_objects.append(Motorbike(detected_object, self))
        elif detected_object.class_type == 15:
            self.image_objects.append(Person(detected_object, self))
        elif detected_object.class_type == 16:
            self.image_objects.append(PottedPlant(detected_object, self))
        elif detected_object.class_type == 17:
            self.image_objects.append(Sheep(detected_object, self))
        elif detected_object.class_type == 18:
            self.image_objects.append(Sofa(detected_object, self))
        elif detected_object.class_type == 19:
            self.image_objects.append(Train(detected_object, self))
        elif detected_object.class_type == 20:
            self.image_objects.append(TVMonitor(detected_object, self))

    def update(self, detection_time, detected_objects, log_file, trace_file, time_step):
        """
        Detected objects are taken into consideration. Previously observed
        image objects are matched into the new detections. If no match
        is found, a new image object is created. Matched image objects are
        updated. Image objects not matched are removed.
        """

        # Predict new coordinates for each image object
        log_file.write("Image objects ({0:d}), predicted new locations:\n".format(len(self.image_objects)))
        for image_object in self.image_objects:
            image_object.predict(time_step)
            log_file.write("---{0:d} {1:s} {2:6.2f} {3:7.2f} {4:7.2f} {5:7.2f} {6:7.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11:.2f} {12:.2f} {13:.2f}\n".format(
                image_object.id, image_object.name, image_object.confidence,
                image_object.x_min, image_object.x_max, \
                image_object.y_min, image_object.y_max, \
                float(image_object.appearance[0]), float(image_object.appearance[1]), \
                float(image_object.appearance[2]), float(image_object.appearance[3]), \
                float(image_object.appearance[4]), float(image_object.appearance[5]), \
                float(image_object.appearance[6]), float(image_object.appearance[7])))

        # Calculate cost matric for Hungarian algorithm (assignment problem)
        log_file.write("Cost matrix for Hungarian algorithm:\n")
        cost = np.zeros((len(detected_objects), len(self.image_objects)))
        detected_object_index = 0
        for detected_object in detected_objects:
            image_object_index = 0
            found = False # Just to decide whether to print a newline into the log
            for image_object in self.image_objects:
                cost[detected_object_index, image_object_index] = \
                    detected_object.distance(image_object)
                log_file.write("{0:7.2f} ".format(cost[detected_object_index, image_object_index]))
                image_object_index += 1
                found = True
            detected_object_index += 1
            if found:
                log_file.write("\n")

        # Find the optimal assigment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Update matched objects
        log_file.write("Optimal assignment:\n")
        match_index = 0
        for row in row_ind:
            i1 = row_ind[match_index]
            i2 = col_ind[match_index]
            detected_object = detected_objects[i1]
            image_object = self.image_objects[i2]
            log_file.write("Detected object {0:d} matched to image object {1:d}\n".format(detected_object.id, \
                           image_object.id))
            distance = cost[i1,i2]
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
            
        # Any image object not matched is removed
        removes = []
        for image_object in self.image_objects:
            if not image_object.matched:
                log_file.write("Image object {0:d} removed:\n".format(image_object.id))

                log_file.write("---{0:s} {1:6.2f} {2:7.2f} {3:7.2f} {4:7.2f} {5:7.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))

                removes.append(image_object)

        # Delete removed objects
        for remove in removes:
            self.image_objects.remove(remove)

        # If no match for a detected object, create a new image object
        for detected_object in detected_objects:
            if not detected_object.matched:
                self.create_image_object(detected_object)
    
                log_file.write("Image object {0:d} created: ".format(next_id.image_object_counter))
    
                log_file.write("{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    CLASS_NAMES[detected_object.class_type], detected_object.confidence,
                    detected_object.x_min, detected_object.x_max, \
                    detected_object.y_min, detected_object.y_max))
