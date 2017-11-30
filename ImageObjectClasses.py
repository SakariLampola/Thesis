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

class ImageObject:
    """
    Generic base class
    """
    name = 'generic'

    height_min, height_mean, height_max = 0.0, nan, inf
    width_min, width_mean, width_max = 0.0, nan, inf
    length_min, length_mean, length_max = 0.0, nan, inf

    velocity_max = inf
    acceleration_max = inf

    def __init__(self, x_min, x_max, y_min, y_max, confidence):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.confidence = confidence
        self.identity = -1

    def bounding_box(self):
        """
        Returns the bounding box of an object: xmin, xmax, ymin, ymax
        """
        return self.x_min, self.x_max, self.y_min, self.y_max

    def center(self):
        """
        Returns the center coordinates of an object: x, y
        """
        return (self.x_max-self.x_min)/2, (self.y_max-self.y_min)/2

    def print_attributes(self):
        """
        Prints the object attributes
        """
        print(self.name, self.x_min, self.x_max, self.y_min, self.y_max, \
              self.confidence)

class Aeroplane(ImageObject):
    """
    Aeroplane - derived class
    """
    name = 'aeroplane'

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
