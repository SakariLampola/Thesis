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

SIMILARITY_DISTANCE = 20 # Max distance for similarity interpretation
RETENTION_TIME = 0.0 # How long image objects are maintained without new detections
CONFIDENFE_LEVEL = 0.0 # How confident we must be to create a new object

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

def get_next_id():
    """
    Generate a unique id
    """
    get_next_id.counter += 1
    return get_next_id.counter

get_next_id.counter = 0

class DetectedObject:
    """
    Raw object detection information
    """
    def __init__(self, time, class_type, x_min, x_max, y_min, y_max, confidence):
        """
        Initialization
        """
        self.time = time
        self.class_type = class_type
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.confidence = confidence
        self.matched = False # By default, not matched with any image object

    def distance(self, time, image_object):
        """
        Calculates Euclidean distance (4 corners) to a predicted image object
        """
        dx_min = abs(self.x_min - image_object.x_min_predicted)
        dx_max = abs(self.x_max - image_object.x_max_predicted)
        dy_min = abs(self.y_min - image_object.y_min_predicted)
        dy_max = abs(self.y_max - image_object.y_max_predicted)

        return (dx_min + dx_max + dy_min + dy_max) / 4.0 # Average of 4 corners

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

    def __init__(self, detected_object):
        """
        Initialization
        """
        self.id = get_next_id()
        self.last_corrected = detected_object.time

        self.confidence = detected_object.confidence
        
        self.track = []
        self.track.append((detected_object.time, detected_object.x_min, \
                           detected_object.x_max, detected_object.y_min, \
                           detected_object.y_max))

        self.x_min = detected_object.x_min
        self.x_max = detected_object.x_max
        self.y_min = detected_object.y_min
        self.y_max = detected_object.y_max

        self.x_min_predicted = nan
        self.x_max_predicted = nan
        self.y_min_predicted = nan
        self.y_max_predicted = nan

    def predict(self, time):
        """
        Predict bounding box coordinates
        """
        if (len(self.track) > 1):
            t_t, x_min_t, x_max_t, y_min_t, y_max_t = self.track[-2]
            self.x_min_predicted = self.x_min + (time-t_t)*(self.x_min-x_min_t)/(time-t_t)
            self.x_max_predicted = self.x_max + (time-t_t)*(self.x_max-x_max_t)/(time-t_t)
            self.y_min_predicted = self.y_min + (time-t_t)*(self.y_min-y_min_t)/(time-t_t)
            self.y_max_predicted = self.y_max + (time-t_t)*(self.y_max-y_max_t)/(time-t_t)
        else:
            self.x_min_predicted = self.x_min
            self.x_max_predicted = self.x_max
            self.y_min_predicted = self.y_min
            self.y_max_predicted = self.y_max
        
    def correct(self, detected_object):
        """
        Correct bounding box coordinates based on detected object
        """
        self.last_corrected = detected_object.time
       
        self.confidence = detected_object.confidence

        self.x_min = detected_object.x_min
        self.x_max = detected_object.x_max
        self.y_min = detected_object.y_min
        self.y_max = detected_object.y_max

        self.track.append((detected_object.time, self.x_min, self.x_max, 
                           self.y_min, self.y_max))

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
    def __init__(self):
        """
        Initialization
        """
        self.image_objects = [] # In the beginning, the world is empty...

    def create_image_object(self, detected_object):
        """
        Create a new object based on detected object information
        """
        if detected_object.class_type == 1:
            self.image_objects.append(Aeroplane(detected_object))
        elif detected_object.class_type == 2:
            self.image_objects.append(Bicycle(detected_object))
        elif detected_object.class_type == 3:
            self.image_objects.append(Bird(detected_object))
        elif detected_object.class_type == 4:
            self.image_objects.append(Boat(detected_object))
        elif detected_object.class_type == 5:
            self.image_objects.append(Bottle(detected_object))
        elif detected_object.class_type == 6:
            self.image_objects.append(Bus(detected_object))
        elif detected_object.class_type == 7:
            self.image_objects.append(Car(detected_object))
        elif detected_object.class_type == 8:
            self.image_objects.append(Cat(detected_object))
        elif detected_object.class_type == 9:
            self.image_objects.append(Chair(detected_object))
        elif detected_object.class_type == 10:
            self.image_objects.append(Cow(detected_object))
        elif detected_object.class_type == 11:
            self.image_objects.append(DiningTable(detected_object))
        elif detected_object.class_type == 12:
            self.image_objects.append(Dog(detected_object))
        elif detected_object.class_type == 13:
            self.image_objects.append(Horse(detected_object))
        elif detected_object.class_type == 14:
            self.image_objects.append(Motorbike(detected_object))
        elif detected_object.class_type == 15:
            self.image_objects.append(Person(detected_object))
        elif detected_object.class_type == 16:
            self.image_objects.append(PottedPlant(detected_object))
        elif detected_object.class_type == 17:
            self.image_objects.append(Sheep(detected_object))
        elif detected_object.class_type == 18:
            self.image_objects.append(Sofa(detected_object))
        elif detected_object.class_type == 19:
            self.image_objects.append(Train(detected_object))
        elif detected_object.class_type == 20:
            self.image_objects.append(TVMonitor(detected_object))

    def update(self, detection_time, detected_objects, log_file, trace_file):
        """
        Detected objects are taken into consideration. Previously observed
        image objects are matched into the new detections. If no correspondance
        is found, a new image object is created. Matched image objects are
        updated. Finally, image objects not updated recently are removed.
        """

        log_file.write("Image object predicted locations:\n")
        print("Image object predicted locations:")
        # Predict new coordinates for each image object
        for image_object in self.image_objects:

            image_object.predict(detection_time)

            log_file.write("---{0:d} {1:s} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f}\n".format(
                image_object.id, image_object.name, image_object.confidence,
                image_object.x_min_predicted, image_object.x_max_predicted, \
                image_object.y_min_predicted, image_object.y_max_predicted))
            print("---{0:d} {1:s} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f}".format(
                image_object.id, image_object.name, image_object.confidence,
                image_object.x_min_predicted, image_object.x_max_predicted, \
                image_object.y_min_predicted, image_object.y_max_predicted))

        # Find matching pairs which are probably the same object in two frames
        log_file.write("Searching image objects for detected objects:\n")
        print("Searching image objects for detected objects:")
        matches = dict()
        for detected_object in detected_objects:

            log_file.write("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                CLASS_NAMES[detected_object.class_type], \
                detected_object.confidence,
                detected_object.x_min, detected_object.x_max, \
                detected_object.y_min, detected_object.y_max))
            print("---{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}".format(
                CLASS_NAMES[detected_object.class_type], \
                detected_object.confidence, \
                detected_object.x_min, detected_object.x_max, \
                detected_object.y_min, detected_object.y_max))

            for image_object in self.image_objects:

                if detected_object.class_type == image_object.class_type: # Must be same type

                    distance = detected_object.distance(detection_time, image_object)

                    log_file.write("   ---   {0:d} {1:s} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f}\n".format( \
                                   image_object.id, image_object.name, image_object.confidence, \
                                   image_object.x_min, image_object.x_max, \
                                   image_object.y_min, image_object.y_max, distance))
                    print("   --- {0:d} {1:s} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f}".format(
                        image_object.id, image_object.name, image_object.confidence, \
                        image_object.x_min, image_object.x_max, \
                        image_object.y_min, image_object.y_max, distance))
                    
                    if distance < SIMILARITY_DISTANCE: # ... and near each other

                        if image_object in matches: # Make sure of the min distance
                            old_object, old_distance = matches[image_object]
                            if distance < old_distance:
                                matches[image_object] = (detected_object, distance)
                                detected_object.matched = True
                                old_object.matched = False
                                log_file.write("      --- Matched (updated)!\n")
                                print("      --- Matched (updated)!")
                        else:
                            matches[image_object] = (detected_object, distance)
                            detected_object.matched = True
                            log_file.write("      --- Matched!\n")
                            print("      --- Matched!")

        # Matched image objects have to be updated and unmatched removed
        for image_object in self.image_objects:

            if image_object in matches: # Update coordinates

                detected_object, distance = matches[image_object]

                log_file.write("Existing image object {0:d} updated:\n".format(image_object.id))
                print("Existing image object {0:d} updated:".format(image_object.id))

                log_file.write("---current:   {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}\n".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))
                print("---current:   {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))

                log_file.write("---predicted: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}\n".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min_predicted, image_object.x_max_predicted, \
                    image_object.y_min_predicted, image_object.y_max_predicted))
                print("---predicted: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min_predicted, image_object.x_max_predicted, \
                    image_object.y_min_predicted, image_object.y_max_predicted))

                log_file.write("---measured:  {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f}\n".format( \
                    CLASS_NAMES[detected_object.class_type], detected_object.confidence, \
                    float(detected_object.x_min), float(detected_object.x_max), \
                    float(detected_object.y_min), float(detected_object.y_max), \
                    distance))
                print("---measured:  {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f}".format( \
                    CLASS_NAMES[detected_object.class_type], detected_object.confidence, \
                    float(detected_object.x_min), float(detected_object.x_max), \
                    float(detected_object.y_min), float(detected_object.y_max), \
                    distance))

                image_object.correct(detected_object)

                log_file.write("---corrected: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}\n".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))
                print("---corrected: {0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}".format(
                    image_object.name, image_object.confidence, \
                    image_object.x_min, image_object.x_max, \
                    image_object.y_min, image_object.y_max))

                fmt = ("{0:.2f},{1:d},{2:.2f},{3:.2f},{4:.2f},{5:.2f},"
                       "{6:.2f},{7:.2f},{8:.2f},{9:.2f},{10:.2f},{11:.2f},"
                       "{12:.2f},{13:.2f},{14:.2f},{1:d}\n")
                trace_file.write(fmt.format(detection_time, \
                                            image_object.id, \
                                            image_object.x_min_predicted, \
                                            image_object.x_max_predicted, \
                                            image_object.y_min_predicted, \
                                            image_object.y_max_predicted, \
                                            detected_object.x_min, \
                                            detected_object.x_max, \
                                            detected_object.y_min, \
                                            detected_object.y_max, \
                                            image_object.x_min, \
                                            image_object.x_max, \
                                            image_object.y_min, \
                                            image_object.y_max, \
                                            distance, \
                                            len(image_object.track)))

            else:

                log_file.write("Image object {0:d} deleted:\n".format(image_object.id))
                print("Image object {0:d} deleted:".format(image_object.id))

                log_file.write("---{0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}\n".format(
                    image_object.name, image_object.confidence, image_object.x_min,
                    image_object.x_max, image_object.y_min, image_object.y_max))
                print("---{0:s} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f}".format(
                    image_object.name, image_object.confidence, image_object.x_min, \
                    image_object.x_max, image_object.y_min, image_object.y_max))

                self.image_objects.remove(image_object)
            

        # New image objects are created based on unmatched detected objects
        for detected_object in detected_objects:
            if detected_object.matched == False:
                if detected_object.confidence > CONFIDENFE_LEVEL:

                    self.create_image_object(detected_object)
    
                    log_file.write("New image object {0:d} created: ".format(get_next_id.counter))
                    print("New image object {0:d} created: ".format(get_next_id.counter))
    
                    log_file.write("{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}\n".format(
                        CLASS_NAMES[detected_object.class_type], detected_object.confidence,
                        detected_object.x_min, detected_object.x_max, \
                        detected_object.y_min, detected_object.y_max))
                    print("{0:s} {1:6.2f} {2:4d} {3:4d} {4:4d} {5:4d}".format(
                        CLASS_NAMES[detected_object.class_type], detected_object.confidence,
                        detected_object.x_min, detected_object.x_max, \
                        detected_object.y_min, detected_object.y_max))
