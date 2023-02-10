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
        return np.mean(speeds)
    
    
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
    
    
    # def hjorthMobilityX(self, coords):
    #     # Extract the time-series data
    #     x = coords[:, 1];
        
    #     # Calculate the mobility
    #     if np.var(x) == 0:
    #         return 10000
    #     return np.sqrt(np.var(np.diff(x)) / np.var(x))
            
    # def hjorthMobilityY(self, coords):
    #     # Extract the time-series data
    #     y = coords[:, 2]
        
    #     # Calculate the mobility
    #     if np.var(y) == 0:
    #         return 10000
    #     return np.sqrt(np.var(np.diff(y)) / np.var(y))
            
    # def hjorthActivityX(self, coords):
    #     # Extract the time-series data
    #     x = coords[:, 1]
    #     # Calculate the activity
    #     return np.var(x)
        
    # def hjorthActivityY(self, coords):
    #     # Extract the time-series data
    #     y = coords[:, 2]
    #     # Calculate the activity
    #     return np.var(y)
    
    # def hjorthComplexityX(self, coords):
    #     # Extract the time-series data
    #     x = coords[:, 1]
    #     # Calculate the complexity
    #     if np.var(np.diff(x)) == 0:
    #         return 10000
    #     return np.sqrt(np.var(np.diff(np.diff(x))) / np.var(np.diff(x)))
        
    # def hjorthComplexityY(self, coords):
    #     # Extract the time-series data
    #     y = coords[:, 2]
    #     # Calculate the complexity
    #     if np.var(np.diff(y)) == 0:
    #         return 10000
    #     return np.sqrt(np.var(np.diff(np.diff(y))) / np.var(np.diff(y)))

    # def hjorthCombinedX(self, coords):
    #     complexity = self.hjorthComplexityX(coords)
    #     if complexity == 0:
    #         return 10000
    #     return self.hjorthMobilityX(coords) * self.hjorthActivityX(coords) / complexity
    
    # def hjorthCombinedY(self, coords):
    #     complexity = self.hjorthComplexityY(coords)
    #     if complexity == 0:
    #         return 10000
    #     return self.hjorthMobilityY(coords) * self.hjorthActivityY(coords) / complexity
    
    # def trackSpeedOverAccel(self, coords):
    #     t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
    #     velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(t)
    #     acceleration = np.diff(velocity) / np.diff(t[:-1])
    #     if np.std(acceleration) == 0:
    #         return 10000
    #     return np.mean(velocity) / np.std(acceleration)
    
    def max_hjorth_parameters(self, coords):
        window_size = 100
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
            t_diff = np.diff(t[start:end])
            mobility[i] = np.sqrt(np.mean(x_diff**2 + y_diff**2) / np.var(t_diff))
            complexity[i] = np.sqrt(np.var(x_diff**2 + y_diff**2) / np.mean(t_diff**2))
            activity[i] = np.var(x[start:end] + y[start:end])
        return np.max(mobility) - np.min(mobility)

    