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
    