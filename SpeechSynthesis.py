# -*- coding: utf-8 -*-
"""
Wrapper class for speech synthesis

Created on Wed Jan 17 11:08:15 2018
@author: Sakari Lampola
"""

import pyttsx3

class SpeechSynthesizer:

    def __init__(self):
        """
        Initialization
        """
        self.engine = pyttsx3.init()

    def say(self, text):
        """
        Say the text
        """
        self.engine.say(text)
        self.engine.runAndWait() # this has to be changed later
