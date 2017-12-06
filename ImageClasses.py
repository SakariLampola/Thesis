# -*- coding: utf-8 -*-
"""
ImageObject class definitions

All attribute units are in SI (m, m/s, m/s**2).
'Height', 'witdth' and 'length' are defined as looking the subject directly
from the front. 'Length' corresponds to 'depth'.
Created on Wed Nov 29 09:08:16 2017
@author: Sakari Lampola
"""

from math import inf, nan, sqrt
import numpy as np

def kalman_filter(mean_1, sigma_1, z, a, c, r, q):
    """
    Kalman filter algorithm
    """
    mean_b = np.dot(a, mean_1)
    sigma_b = np.dot(np.dot(a, sigma_1), a.T) + r
    k1 = np.dot(sigma_b, c.T)
    k2 = np.linalg.inv(np.dot(np.dot(c, sigma_1), c.T) + q)
    k = np.dot(k1, k2)
    mean = mean_b + np.dot(k, z - np.dot(c, mean_b))
    s1 = np.dot(k, c)
    n1, n2 = np.shape(s1)
    s1 = np.eye(n1) - s1
    sigma = np.dot(s1, sigma_b)
    return mean, sigma

LOC_R = np.zeros((6, 6)) # Covariance matrix for location state noise
LOC_R[0, 0] = 1.0
LOC_R[1, 1] = 1.0
LOC_R[2, 2] = 1.0
LOC_R[3, 3] = 1.0
LOC_R[4, 4] = 1.0
LOC_R[5, 5] = 1.0

LOC_Q = np.zeros((2, 2)) # Covariance matrix for location measurement noise
LOC_Q[0, 0] = 100.0
LOC_Q[1, 1] = 100.0

LOC_SIGMA_ALFA = 1.0 # Initial location covariance
LOC_SIGMA_BETA = 1.0 # Initial velocity covariance
LOC_SIGMA_GAMMA = 1.0 # Initial acceleration covariance

SIZE_R = np.zeros((6, 6)) # Covariance matrix for size state noise
SIZE_R[0, 0] = 1.0
SIZE_R[1, 1] = 1.0
SIZE_R[2, 2] = 1.0
SIZE_R[3, 3] = 1.0
SIZE_R[4, 4] = 1.0
SIZE_R[5, 5] = 1.0

SIZE_Q = np.zeros((2, 2)) # Covariance matrix for size measurement noise
SIZE_Q[0, 0] = 1000.0
SIZE_Q[1, 1] = 1000.0

SIZE_SIGMA_ALFA = 1.0 # Initial size covariance
SIZE_SIGMA_BETA = 1.0 # Initial (size) velocity covariance
SIZE_SIGMA_GAMMA = 1.0 # Initial (size) acceleration covariance

SIMILARITY_DISTANCE = 100 # Max distance for similarity interpretation
RETENTION_TIME = 0.5 # How long objects are maintained without new measurements

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

def get_next_id():
    """
    Generate unique id
    """
    get_next_id.counter += 1
    return get_next_id.counter

get_next_id.counter = 0

class Measurement:
    """
    Raw object detection information
    """
    def __init__(self, idx, x_min, x_max, y_min, y_max, confidence):
        self.idx = idx
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.confidence = confidence

    def distance(self, image_object):
        """
        Calculates Euclidean distance to image object
        """
        xd = (self.x_min + self.x_max)/2.0 - image_object.loc_mean[0]
        yd = (self.y_min + self.y_max)/2.0 - image_object.loc_mean[1]
        return sqrt(xd*xd+yd*yd)

    def bounding_box(self):
        """
        Get the bounding box
        """
        return self.x_min, self.x_max, self.y_min, self.y_max


class ImageObject:
    """
    Generic base class
    """
    name = 'generic'
    class_type = 0

    height_min, height_mean, height_max = 0.0, nan, inf
    width_min, width_mean, width_max = 0.0, nan, inf
    length_min, length_mean, length_max = 0.0, nan, inf

    velocity_max = inf
    acceleration_max = inf

    def __init__(self, time, x_min, x_max, y_min, y_max, confidence):

        self.id = get_next_id()
        
        self.last_update = time

        self.loc_mean = np.array([(x_min+x_max)/2.0, (y_min+y_max)/2.0, \
                                  0.0, 0.0, 0.0, 0.0])
        self.loc_sigma = np.zeros((6, 6))
        self.loc_sigma[0, 0] = LOC_SIGMA_ALFA
        self.loc_sigma[1, 1] = LOC_SIGMA_ALFA
        self.loc_sigma[2, 2] = LOC_SIGMA_BETA
        self.loc_sigma[3, 3] = LOC_SIGMA_BETA
        self.loc_sigma[4, 4] = LOC_SIGMA_GAMMA
        self.loc_sigma[5, 5] = LOC_SIGMA_GAMMA

        self.size_mean = np.array([x_max - x_min, y_max - y_min, \
                                  0.0, 0.0, 0.0, 0.0])
        self.size_sigma = np.zeros((6, 6))
        self.size_sigma[0, 0] = SIZE_SIGMA_ALFA
        self.size_sigma[1, 1] = SIZE_SIGMA_ALFA
        self.size_sigma[2, 2] = SIZE_SIGMA_BETA
        self.size_sigma[3, 3] = SIZE_SIGMA_BETA
        self.size_sigma[4, 4] = SIZE_SIGMA_GAMMA
        self.size_sigma[5, 5] = SIZE_SIGMA_GAMMA
        
        self.confidence = confidence

    def bounding_box(self):
        """
        Get the bounding box
        """
        x_min = int(self.loc_mean[0] - self.size_mean[0]/2.0)
        x_max = int(self.loc_mean[0] + self.size_mean[0]/2.0)
        y_min = int(self.loc_mean[1] - self.size_mean[1]/2.0)
        y_max = int(self.loc_mean[1] + self.size_mean[1]/2.0)
        return x_min, x_max, y_min, y_max
    
    def distance(self, another_object):
        """
        Calculates Euclidean distance to another object
        """
        xd = self.loc_mean[0]-another_object.loc_mean[0]
        yd = self.loc_mean[1]-another_object.loc_mean[1]
        return sqrt(xd*xd+yd*yd)

    def update(self, time, bounding_box, confidence, trace_file):
        """
        Update location and size based on new measurement
        """
        delta = time - self.last_update
        self.last_update = time

        x_min = bounding_box[0]
        x_max = bounding_box[1]
        y_min = bounding_box[2]
        y_max = bounding_box[3]

        loc_z = np.array([(x_min+x_max)/2, (y_min+y_max)/2])
        loc_a = np.eye((6))
        loc_a[0, 2] = delta
        loc_a[1, 3] = delta
        loc_a[2, 4] = delta
        loc_a[3, 5] = delta
        loc_c = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
                          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

        size_z = np.array([x_max - x_min, y_max - y_min])
        size_a = np.eye((6))
        size_a[0, 2] = delta
        size_a[1, 3] = delta
        size_a[2, 4] = delta
        size_a[3, 5] = delta
        size_c = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
                          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

        self.loc_mean, self.loc_sigma = \
            kalman_filter(self.loc_mean, self.loc_sigma, loc_z, loc_a, loc_c, \
                          LOC_R, LOC_Q)
        
        self.size_mean, self.size_sigma = \
            kalman_filter(self.size_mean, self.size_sigma, size_z, size_a, size_c, \
                          SIZE_R, SIZE_Q)

        if self.id == 4:
            fmt = ("{0:.2f},{1:.2f},{2:.2f},"
                   "{3:.2f},{4:.2f},"
                   "{5:.2f},{6:.2f},"
                   "{7:.2f},{8:.2f},"
                   "{9:.2f},{10:.2f},"
                   "{11:.2f},{12:.2f},"
                   "{13:.2f},{14:.2f},"
                   "{15:.2f},{16:.2f}"
                   "\n"
                )
            trace_file.write(fmt.format(time, loc_z[0], loc_z[1], \
                                        self.loc_mean[0], self.loc_mean[1], \
                                        self.loc_mean[2], self.loc_mean[3], \
                                        self.loc_mean[4], self.loc_mean[5], \
                                        size_z[0], size_z[1], \
                                        self.size_mean[0], self.size_mean[1], \
                                        self.size_mean[2], self.size_mean[3], \
                                        self.size_mean[4], self.size_mean[5]))


class Aeroplane(ImageObject):
    """
    Aeroplane - derived class
    """
    name = 'aeroplane'
    class_type = 1

    height_min = 1.5
    height_mean = 2.0
    height_max = 6.0

    width_min = 2.0
    width_mean = 20.0
    width_max = 80.0

    length_min = 3.0
    length_mean = 20.0
    length_max = 80.0

    velocity_max = 300.0
    acceleration_max = 10.0

class Bicycle(ImageObject):
    """
    Bicycle - derived class
    """
    name = 'bicycle'
    class_type = 2

    height_min = 0.6
    height_mean = 1.1
    height_max = 1.4

    width_min = 0.4
    width_mean = 0.6
    width_max = 0.8

    length_min = 1.2
    length_mean = 1.5
    length_max = 1.7

    velocity_max = 13.0
    acceleration_max = 1.0

class Bird(ImageObject):
    """
    Bird - derived class
    """
    name = 'bird'
    class_type = 3

    height_min = 0.1
    height_mean = 0.25
    height_max = 2.8

    width_min = 0.05
    width_mean = 0.5
    width_max = 1.2

    length_min = 0.1
    length_mean = 0.2
    length_max = 1.2

    velocity_max = 80.0
    acceleration_max = 100.0

class Boat(ImageObject):
    """
    Boat - derived class
    """
    name = 'boat'
    class_type = 4

    height_min = 1.0
    height_mean = 2.0
    height_max = 20.0

    width_min = 0.6
    width_mean = 3.0
    width_max = 60.0

    length_min = 1.5
    length_mean = 3.0
    length_max = 350.0

    velocity_max = 20.0
    acceleration_max = 5.0

class Bottle(ImageObject):
    """
    Bottle - derived class
    """
    name = 'bottle'
    class_type = 5

    height_min = 0.1
    height_mean = 0.25
    height_max = 0.5

    width_min = 0.05
    width_mean = 0.1
    width_max = 0.3

    length_min = 0.05
    length_mean = 0.1
    length_max = 0.3

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class Bus(ImageObject):
    """
    Bus - derived class
    """
    name = 'bus'
    class_type = 6

    height_min = 2.5
    height_mean = 2.8
    height_max = 5.0

    width_min = 2.3
    width_mean = 2.5
    width_max = 2.8

    length_min = 6.0
    length_mean = 15.0
    length_max = 30.0

    velocity_max = 30.0
    acceleration_max = 2.0

class Car(ImageObject):
    """
    Car - derived class
    """
    name = 'car'
    class_type = 7

    height_min = 1.3
    height_mean = 1.5
    height_max = 2.0

    width_min = 1.6
    width_mean = 1.8
    width_max = 2.0

    length_min = 3.5
    length_mean = 4.3
    length_max = 5.5

    velocity_max = 50.0
    acceleration_max = 10.0

class Cat(ImageObject):
    """
    Cat - derived class
    """
    name = 'cat'
    class_type = 8

    height_min = 0.2
    height_mean = 0.3
    height_max = 0.4

    width_min = 0.1
    width_mean = 0.15
    width_max = 0.2

    length_min = 0.3
    length_mean = 0.4
    length_max = 0.5

    velocity_max = 13.0
    acceleration_max = 10.0

class Chair(ImageObject):
    """
    Chair - derived class
    """
    name = 'chair'
    class_type = 9

    height_min = 0.5
    height_mean = 1.0
    height_max = 1.5

    width_min = 0.5
    width_mean = 0.6
    width_max = 0.7

    length_min = 0.5
    length_mean = 0.6
    length_max = 0.7

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class Cow(ImageObject):
    """
    Cow - derived class
    """
    name = 'cow'
    class_type = 10

    height_min = 1.0
    height_mean = 1.3
    height_max = 1.5

    width_min = 0.5
    width_mean = 0.6
    width_max = 0.7

    length_min = 1.8
    length_mean = 2.3
    length_max = 2.8

    velocity_max = 10.0
    acceleration_max = 2.0

class DiningTable(ImageObject):
    """
    Dining table - derived class
    """
    name = 'dining table'
    class_type = 11

    height_min = 0.7
    height_mean = 0.75
    height_max = 0.9

    width_min = 0.5
    width_mean = 1.0
    width_max = 1.5

    length_min = 0.5
    length_mean = 2.0
    length_max = 5.0

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class Dog(ImageObject):
    """
    Dog - derived class
    """
    name = 'dog'
    class_type = 12

    height_min = 0.2
    height_mean = 0.35
    height_max = 0.8

    width_min = 0.15
    width_mean = 0.2
    width_max = 0.3

    length_min = 0.3
    length_mean = 0.6
    length_max = 1.3

    velocity_max = 13
    acceleration_max = 10

class Horse(ImageObject):
    """
    Horse - derived class
    """
    name = 'horse'
    class_type = 13

    height_min = 1.2
    height_mean = 1.6
    height_max = 1.8

    width_min = 0.3
    width_mean = 0.45
    width_max = 0.6

    length_min = 2.0
    length_mean = 2.4
    length_max = 3.0

    velocity_max = 13
    acceleration_max = 4

class Motorbike(ImageObject):
    """
    Motorbike - derived class
    """
    name = 'motorbike'
    class_type = 14

    height_min = 0.6
    height_mean = 1.1
    height_max = 1.4

    width_min = 0.4
    width_mean = 0.6
    width_max = 0.8

    length_min = 1.2
    length_mean = 1.5
    length_max = 1.7

    velocity_max = 50.0
    acceleration_max = 10.0

class Person(ImageObject):
    """
    Person - derived class
    """
    name = 'person'
    class_type = 15

    height_min = 1.0
    height_mean = 1.7
    height_max = 2.0

    width_min = 0.5
    width_mean = 0.6
    width_max = 0.7

    length_min = 0.25
    length_mean = 0.35
    length_max = 0.45

    velocity_max = 10.0
    acceleration_max = 3.0

class PottedPlant(ImageObject):
    """
    Potted plant - derived class
    """
    name = 'potted plant'
    class_type = 16

    height_min = 0.2
    height_mean = 0.35
    height_max = 1.0

    width_min = 0.15
    width_mean = 0.25
    width_max = 0.4

    length_min = 0.15
    length_mean = 0.25
    length_max = 0.4

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class Sheep(ImageObject):
    """
    Sheep - derived class
    """
    name = 'sheep'
    class_type = 17

    height_min = 0.8
    height_mean = 1.2
    height_max = 1.4

    width_min = 0.25
    width_mean = 0.35
    width_max = 0.45

    length_min = 1.0
    length_mean = 1.3
    length_max = 1.5

    velocity_max = 10
    acceleration_max = 6

class Sofa(ImageObject):
    """
    Sofa - derived class
    """
    name = 'sofa'
    class_type = 18

    height_min = 0.7
    height_mean = 1.0
    height_max = 1.4

    width_min = 0.7
    width_mean = 1.0
    width_max = 1.5

    length_min = 1.5
    length_mean = 2.0
    length_max = 4.0

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class Train(ImageObject):
    """
    Train - derived class
    """
    name = 'train'
    class_type = 19

    height_min = 2.5
    height_mean = 3.0
    height_max = 5.0

    width_min = 2.5
    width_mean = 3.1
    width_max = 3.5

    length_min = 25.0
    length_mean = 150.0
    length_max = 1000.0

    velocity_max = 50.0
    acceleration_max = 1.0

class TVMonitor(ImageObject):
    """
    TV Monitor - derived class
    """
    name = 'tv monitor'
    class_type = 20

    height_min = 0.2
    height_mean = 0.5
    height_max = 1.2

    width_min = 0.3
    width_mean = 0.8
    width_max = 2.0

    length_min = 0.04
    length_mean = 0.1
    length_max = 0.5

    velocity_max = 10.0 # same as Person
    acceleration_max = 3.0 # same as Person

class ImageWorld:
    """
    Updatable list of objects with internal state and ability to forecast
    """
    def __init__(self):
        self.world_objects = [] # In the beginning, the world is empty
        
    def insert_object(self, time, measurement):
        """
        Insert a new object based on measurement into the world
        """
        idx = measurement.idx
        x_min = measurement.x_min
        x_max = measurement.x_max
        y_min = measurement.y_min
        y_max = measurement.y_max
        confidence = measurement.confidence
        
        if idx == 1:
            self.world_objects.append(Aeroplane(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 2:
            self.world_objects.append(Bicycle(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 3:
            self.world_objects.append(Bird(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 4:
            self.world_objects.append(Boat(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 5:
            self.world_objects.append(Bottle(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 6:
            self.world_objects.append(Bus(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 7:
            self.world_objects.append(Car(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 8:
            self.world_objects.append(Cat(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 9:
            self.world_objects.append(Chair(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 10:
            self.world_objects.append(Cow(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 11:
            self.world_objects.append(DiningTable(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 12:
            self.world_objects.append(Dog(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 13:
            self.world_objects.append(Horse(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 14:
            self.world_objects.append(Motorbike(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 15:
            self.world_objects.append(Person(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 16:
            self.world_objects.append(PottedPlant(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 17:
            self.world_objects.append(Sheep(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 18:
            self.world_objects.append(Sofa(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 19:
            self.world_objects.append(Train(time, x_min, x_max, y_min, y_max, confidence))
        elif idx == 20:
            self.world_objects.append(TVMonitor(time, x_min, x_max, y_min, y_max, confidence))
        

    def update(self, time, detected_objects, log_file, trace_file):
        """
        New object measurements are taken into consideration. Previously
        observed objects are matched into the new measurements.
        """
        # find matching pairs which are probably the same object in two frames
        matches = dict()
        detected_object_index = -1
        for detected_object in detected_objects:
            detected_object_index = detected_object_index + 1
            world_object_index = -1
            for world_object in self.world_objects:
                world_object_index = world_object_index + 1
                distance = detected_object.distance(world_object)
                if (detected_object.idx == world_object.class_type): # got to be same type
                    if (distance < SIMILARITY_DISTANCE): # and near each other
                        if detected_object_index in matches: # make sure of the min distance
                            old_index, old_distance = matches[detected_object_index]
                            if distance < old_distance:
                                matches[detected_object_index] = (world_object_index, distance)
                        else:
                            matches[detected_object_index] = (world_object_index, distance)

        # deside what to do with observed objects        
        detected_object_index = -1
        for detected_object in detected_objects:
            detected_object_index = detected_object_index + 1
            if detected_object_index in matches: # update existing object
                old_index, old_distance = matches[detected_object_index]

                log_file.write("Existing object {0:d} updated.\n".\
                               format(self.world_objects[old_index].id))
                x_min, x_max, y_min, y_max = \
                    self.world_objects[old_index].bounding_box()
                log_file.write("---old: {0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    self.world_objects[old_index].name, \
                    self.world_objects[old_index].confidence, \
                    x_min, x_max, y_min, y_max))

                x_min, x_max, y_min, y_max = detected_object.bounding_box()
                log_file.write("---new: {0:d} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    detected_object.idx, detected_object.confidence,
                    x_min, x_max, y_min, y_max))

                self.world_objects[old_index].update(time, \
                                  detected_object.bounding_box(), \
                                  detected_object.confidence, trace_file)
            else: # create a new one
                log_file.write("New object {0:d} created.\n".format(detected_object.idx))
                x_min, x_max, y_min, y_max = detected_object.bounding_box()
                log_file.write("---{0:d} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                    detected_object.idx, detected_object.confidence,
                    x_min, x_max, y_min, y_max))

                self.insert_object(time, detected_object)

        # remove objects without fresh measurements
        removes = []
        remove_index = -1
        for world_object in self.world_objects:
            remove_index = remove_index + 1
            if time - world_object.last_update > RETENTION_TIME:
                removes.append(remove_index)
        for remove_index in removes:
            log_file.write("Object {0:d} deleted.\n".\
                           format(self.world_objects[remove_index].id))
            x_min, x_max, y_min, y_max = self.world_objects[remove_index].bounding_box()
            log_file.write("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                self.world_objects[remove_index].name, \
                self.world_objects[remove_index].confidence, \
                x_min, x_max, y_min, y_max))
            
            del self.world_objects[remove_index]
            