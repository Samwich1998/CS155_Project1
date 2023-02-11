#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Need to Install on the Anaconda Prompt:
    $ pip install pyexcel
"""


# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General modules
import os
from datetime  import datetime

import csv
import numpy as np

# -------------------------------------------------------------------------- #
# ---------------------------- Compile Features ---------------------------- #

def compileFeatures(track_data, TYPE):
    print("\nCompile Features for " + TYPE.capitalize() + "ing Data")
    # Specify output filename
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILENAME = f"./Data/Input Data/{TYPE}_features_{TIMESTAMP}.csv"
    
    # Compile feature extraction methods
    extractionClass = featureExtractionProtocols()
    FEATURE_LIST = [method for method in dir(extractionClass) if callable(getattr(extractionClass, method)) and not method.startswith("__")]
    
    # Generate the feature csv
    header = ['uid', 'label']
    for featfunc in FEATURE_LIST:
        featfunc = getattr(extractionClass, featfunc)
        header.append(featfunc.__name__)
    
    features = []
    
    track_uids = track_data.keys()
    for uid in track_uids:
        curr_row = {
            'uid': uid,
            'label': track_data[uid]['label']
        }
        
        for featfunc in FEATURE_LIST:
            featfunc = getattr(extractionClass, featfunc)
            curr_row[featfunc.__name__] = featfunc(np.array(track_data[uid]['txy']))
        
        features.append(curr_row)
    
    with open(OUTPUT_FILENAME, 'w') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        for r in features:
            writer.writerow(r)
    
    print("\tWritten to:", OUTPUT_FILENAME)

# -------------------------------------------------------------------------- #
# ---------------------- Feature Extraction Protocols ---------------------- #


def hjorthParameters(coords):
    window_size = 60
    t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
    N = len(t)
    mobility = np.zeros(N)
    complexity = np.zeros(N)
    activity = np.zeros(N)
    for i in range(N):
        start = max(0, i - window_size)
        end = min(N, i + window_size)
        x_diff = np.diff(x[start:end])
        y_diff = np.diff(y[start:end])
        if len(x_diff) == 0:
            mobility[i] = np.nan
            complexity[i] = np.nan
            activity[i] = np.nan
        else:
            mobility[i] = np.sqrt(np.nanmean(x_diff**2 + y_diff**2))
            complexity[i] = np.sqrt(np.var(np.diff(x_diff**2 + y_diff**2)))
            activity[i] = np.var(x[start:end] + y[start:end])
    return activity, complexity, mobility


from scipy.signal import savgol_filter

def curvature(x, y):
    num = len(x)/2
    oddInt = int(num) if int(num) % 2 != 0 else int(num) + 1
    
    y = savgol_filter(y, max(3, oddInt), 2, mode='mirror')    
    # first, calculate the derivative of the y-coordinate with respect to the x-coordinate
    dy_dx = np.gradient(y, x)
    
    # smooth the first derivative using a Savitzky-Golay filter
    
    dy_dx_smooth = savgol_filter(dy_dx, max(3, oddInt), 2, mode='mirror')    
    
    # calculate the second derivative of the y-coordinate with respect to the x-coordinate
    d2y_dx2 = np.gradient(dy_dx_smooth, x)
    
    # calculate the curvature
    curvature = np.abs(d2y_dx2) / (1 + dy_dx_smooth**2)**1.5
    
    return curvature


def count_near_linear_points(x, y, threshold):
    points = np.column_stack((x, y))
    count = 0
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x1, y1 = np.array(points[i], dtype=np.float64)
            x2, y2 = np.array(points[j], dtype=np.float64)
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = 0
            if abs(slope) <= threshold:
                count += 1
    return count


class featureExtractionProtocols:
        
    def mean_step_speed(self, coords):
        """Mean step speed of the entire track.
        
        The average per-step speed. Basically the average of distances between points adjacent in time.
        
        Returns
        -------
        float
            The average step speed.
        """
    
        speeds = []
    
        for i in range(1, coords.shape[0]):
            # Previous coordinate location
            prev = coords[i-1, 1:]
            # Current coordinate location
            curr = coords[i, 1:]
            
            # Speed in pixels per frame
            curr_speed = np.linalg.norm(curr - prev)
            
            # Accumulate per-step speeds into a list
            speeds.append(curr_speed)
        
        # Return the average of the speeds
        return np.nanmean(speeds)
    
    
    def stddev_step_speed(self, coords):
        """Standard deviation of the step speed of the entire track.
        
        The standard deviation of the per-step speed.
        
        Returns
        -------
        float
            The stddev of the step speed.
        """
    
        speeds = []
    
        for i in range(1, coords.shape[0]):
            # Previous coordinate location
            prev = coords[i-1, 1:]
            # Current coordinate location
            curr = coords[i, 1:]
            
            # Speed in pixels per frame
            curr_speed = np.linalg.norm(curr - prev)
            
            # Accumulate per-step speeds into a list
            speeds.append(curr_speed)
        
        # Return the standard deviation of the speeds
        return np.std(speeds)
    
    
    def track_length(self, coords):
        """Standard deviation of the step speed of the entire track.
        
        The standard deviation of the per-step speed.
        
        Returns
        -------
        float
            The length of the entire track.
        """
    
        lengths = []
    
        for i in range(1, coords.shape[0]):
            # Previous coordinate location
            prev = coords[i-1,1:]
            # Current coordinate location
            curr = coords[i,1:]
            
            # Speed in pixels per frame
            step_length = np.linalg.norm(curr - prev)
            
            # Accumulate per-step speeds into a list
            lengths.append(step_length)
        
        # Return the sum of the lengths
        return np.sum(lengths)
    
    
    def e2e_distance(self, coords):
        """End-to-end distance of the track.
        
        The distance from the start and the end of the given track.
        
        Returns
        -------
        float
            The end-to-end distance of the entire track.
        """
        
        # Start and end of the track
        start = coords[0, 1:]
        end = coords[-1, 1:]
        
        # Return the distance
        return np.linalg.norm(end-start)
    
    
    def duration(self, coords):
        """Duration of the track.
        
        The time duration of the track.
        
        Returns
        -------
        int
            The end-to-end duration of the entire track.
        """
        
        # Start and end times of the track
        start_t = coords[0, 0]
        end_t = coords[-1, 0]
        
        # Return the difference
        return end_t - start_t
    
    
    ######################################
    # Implement your own features below! #
    ######################################
    def direction_std(self, coords):
        """Name of the Feature
        
        A short description of the feature goes here. Equations can be useful.
        
        Parameters
        ----------
        coords: array
            A numpy array containing the (t, x, y) coordinates of the track.
        
        Returns
        -------
        float
            The feature value for the entire array.
        
        """
        dir_x = []
        dir_y = []
    
        for i in range(1, coords.shape[0]):
            # Previous coordinate location
            prev = coords[i-1, 1:]
            # Current coordinate location
            curr = coords[i, 1:]
            
            # Speed in pixels per frame
            dir_x.append(curr[0]-prev[0])
            dir_y.append(curr[1]-prev[1])
        
        # Return the standard deviation of the speeds
        return np.sqrt(np.var(dir_x)+np.var(dir_y))
    
    
    
    def hjorthRatio(self, coords):
        activity, complexity, mobility = hjorthParameters(coords)
        
        ratio = activity*complexity/(mobility +1e-10)
        ratio = ratio[0:20]
        
        if not all(np.isnan(ratio)):
            return np.nanmean(ratio)
        else:
            return 0
    
    def hjorthRatio2(self, coords):
        activity, complexity, mobility = hjorthParameters(coords)
        
        ratio = activity/(mobility +1e-10)
        curvatureAC = curvature(coords[:, 0], ratio)[0:20]
        
        if not all(np.isnan(curvatureAC)):
            return np.nanmean(curvatureAC)
        else:
            return 0
    
    
    def hjorthRatio3(self, coords):
        activity, complexity, mobility = hjorthParameters(coords)
        
        ratio = activity/(mobility +1e-10)
        
        num = count_near_linear_points(coords[:, 0], ratio, 1)
        return num if not np.isnan(num) else 0
    
    
    
    
    # Unsure if the ones below are good... may be bad
    ######################################
    
    # def trackSpeedOverAccel(self, coords):
    #     t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
    #     velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(t)
    #     acceleration = np.diff(velocity) / np.diff(t[:-1])
    #     if np.std(acceleration) == 0:
    #         return 10000
    #     return np.nanmean(velocity) / np.std(acceleration)
    


    
    
    
    # def max_time_stationary(self, coords):        
    #     timeSinceLastMovement = 0
    #     maxTimeSinceLastMovement = 0
    #     for i in range(1, coords.shape[0]):
    #         # Previous coordinate location
    #         prev = coords[i-1,1:]
    #         # Current coordinate location
    #         curr = coords[i,1:]
            
    #         # Speed in pixels per frame
    #         step_length = np.linalg.norm(curr - prev)
            
    #         if step_length == 0:
    #             timeSinceLastMovement += 1
    #             maxTimeSinceLastMovement = max(maxTimeSinceLastMovement, timeSinceLastMovement)
    #         else:
    #             timeSinceLastMovement = 0

    #     # Return the sum of the lengths
    #     return maxTimeSinceLastMovement
    