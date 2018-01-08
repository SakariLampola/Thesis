# -*- coding: utf-8 -*-
"""
Analyze trace file

USAGE
python AnalyzeTrace.py

Created on Sat Jan 06 13:44:02 2018
@author: Sakari Lampola
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('trace.txt')

def detected_object_hue_distribution(id):
    object_data = data.loc[data['id']==id]
    dd1 = pd.DataFrame({'Hue': 'f1', 'Fraction' :object_data['do_c1']})
    dd2 = pd.DataFrame({'Hue': 'f2', 'Fraction' :object_data['do_c2']})
    dd3 = pd.DataFrame({'Hue': 'f3', 'Fraction' :object_data['do_c3']})
    dd4 = pd.DataFrame({'Hue': 'f4', 'Fraction' :object_data['do_c4']})
    dd5 = pd.DataFrame({'Hue': 'f5', 'Fraction' :object_data['do_c5']})
    dd6 = pd.DataFrame({'Hue': 'f6', 'Fraction' :object_data['do_c6']})
    dd7 = pd.DataFrame({'Hue': 'f7', 'Fraction' :object_data['do_c7']})
    dd8 = pd.DataFrame({'Hue': 'f8', 'Fraction' :object_data['do_c8']})
    dd = pd.concat([dd1, dd2, dd3, dd4, dd5, dd6, dd7, dd8])
    sns.boxplot(x="Hue", y="Fraction", data=dd)
    
def boundingbox_movement(id):
    object_data = data[['x_min_m','x_max_m','y_min_m','y_max_m']].loc[data['id']==id]
    object_data.plot()    

def boundingbox_movement_diff(id):
    object_data = data[['x_min_m','x_max_m','y_min_m','y_max_m']].loc[data['id']==id]
    object_data = object_data.diff(1)
    object_data = object_data.dropna()
    plt.figure(1)
    sns.pairplot(object_data)
    plt.figure(2)
    object_data.plot()

boundingbox_movement_diff(161)
