# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

sns.set(style="whitegrid", color_codes=True)

# constants
td = 1.0/25.0 # some videos are 1.0/30.0, but for the sake of simplicity all fps are assumed to be the same
minimum_frames = 75 # less than 3 seconds short bursts are filtered away
forecast_horizon = 10 # how many steps ahead we need to forecast

trace_files = ['ImageObjectKalmanFilteringTraceCar.txt','ImageObjectKalmanFilteringTraceCalf.txt',\
              'ImageObjectKalmanFilteringTraceBird.txt','ImageObjectKalmanFilteringTraceSailingBoat.txt',\
              'ImageObjectKalmanFilteringTraceSofa.txt']
#trace_files = ['ImageObjectKalmanFilteringTraceCar.txt']

corners = ['x_min_m', 'x_max_m', 'y_min_m', 'y_max_m']
#corners = ['x_min_m']

# System matrices
A = np.array([[1.0, td],[0.0, 1.0]]) # state equation matrix
C = np.array([[1.0, 0.0]]) # measurement matrix

def opt_callback(x):
    print(x)

def kalman_filter(mu, sigma, data, R, Q):
    # Storage area for measurements and filtered values
    measurement = np.array([data['l'].iloc[0]])
    filtered_location = np.array([data['l'].iloc[0]])
    filtered_velocity = np.array([0.0])
    # Let's filter through the signal
    length,width = data.shape
    for i in range(1, length): # note '1': starting from the second value
        mu = A.dot(mu)
        sigma = A.dot(sigma).dot(A.T) + R
        K = sigma.dot(C.T).dot(np.linalg.inv(C.dot(sigma).dot(C.T) + Q))
        mu = mu + K.dot(data['l'].iloc[i]-C.dot(mu))
        sigma = (np.eye(2)-K.dot(C)).dot(sigma)
        # save the values:
        measurement = np.vstack((measurement, [data['l'].iloc[i]]))
        filtered_location = np.vstack((filtered_location, mu[0]))
        filtered_velocity = np.vstack((filtered_velocity, mu[1]))    
    return measurement, filtered_location, filtered_velocity

def cost_function(x):
    r1, r2, q, alfa, beta = x[0], x[1], x[2], x[3], x[4] 
    frames_total = 0.0
    error_total = 0.0
    # Kalman filter constant matrices
    R = np.array([[r1, 0.0],[0.0, r2]]) # state equation variances
    Q = np.array([q]) # measurent variance
    for trace_file in trace_files:
        data = pd.read_csv(trace_file)
        id_max = data['id'].max()
        for image_object_id in range(1, id_max+1):
#        for image_object_id in range(32, 33):
            image_object_data = data.loc[data['id']==image_object_id]
            length,width = image_object_data.shape
            if (length > minimum_frames):
                image_object_data = data.loc[data['id']==image_object_id]
                for corner in corners:
                    corner_data = pd.DataFrame({'time':image_object_data['time'], 'l' : image_object_data[corner]})
                    corner_data.index = corner_data['time']
                    del corner_data['time']
                    # Kalman filter initial values
                    mu = np.array([[corner_data['l'].iloc[0]],[0.0]])
                    sigma = np.array([[alfa, 0],[0.0, beta]])
                    measurement, filtered_location, filtered_velocity = kalman_filter(mu, sigma, corner_data, R, Q)
                    forecast_error = np.array([0.0]) # first value with no forecast
                    for i in range(1, length-forecast_horizon): # note '1': starting from the second value
                        current_error = 0.0
                        mu = np.array([filtered_location[i],[filtered_velocity[i]]])
                        for j in range(i, i+forecast_horizon):
                            mu = A.dot(mu)
                            current_error += np.abs((mu[0]-filtered_location[j]))
                        forecast_error = np.vstack((forecast_error, current_error / forecast_horizon))
                    mean_error = np.mean(forecast_error)
                    error_total += mean_error * length
                    frames_total += length
                    
    return error_total / frames_total

#r1_values = [1.0, 10.0, 100.0, 1000.0]
#r2_values = [1.0, 10.0, 100.0, 1000.0]
#q_values = [1.0, 10.0, 100.0, 1000.0]
#alfa_values = [1.0, 10.0, 100.0, 1000.0]
#beta_values = [1.0, 10.0, 100.0, 1000.0]
#
#cost_min = np.inf
#r1_min = 0.0
#r2_min = 0.0
#q_min = 0.0
#alfa_min = 0.0
#beta_min = 0.0
#
#for r1 in r1_values:
#    for r2 in r2_values:
#        for q in q_values:
#            for alfa in alfa_values:
#                for beta in beta_values:
#                    cost = cost_function(np.array([r1, r2, q, alfa, beta]))
#                    if cost < cost_min:
#                        cost_min = cost
#                        r1_min = r1
#                        r2_min = r2
#                        q_min = q
#                        alfa_min = alfa
#                        beta_min = beta
#                        print('New optimum!')
#                    print(r1, r2, q, alfa, beta, cost)

x0 = np.array([1.0, 1.0, 1000.0, 1000.0, 1000.0])
print(cost_function(x0))
bnds = ((0, None), (0, None), (0, None), (0, None), (0, None))
res = minimize(cost_function, x0, method='SLSQP', callback=opt_callback, bounds=bnds)
print('Optimum:', res.x)
print(cost_function(res.x))                      