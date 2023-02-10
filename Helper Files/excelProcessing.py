#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Need to Install on the Anaconda Prompt:
    $ pip install pyexcel
"""


# Basic Modules
import os
import json
import numpy as np
# Read/Write to Excel
import csv

class processFiles:
    
    def extractFeatures(self, filePath):
        print("Extracting data from file:", filePath)
        
        fileData = []
        # reading csv file
        with open(filePath, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
             
            # extracting field names through first row
            featureNames = np.array(next(csvreader)[2:])
         
            # extracting each data row one by one
            for row in csvreader:
                fileData.append(row)
         
            # get total number of rows
            print("\tTotal no. of rows: %d"%(csvreader.line_num))
        fileData = np.array(fileData)
        
        
        Y = np.array(fileData[:, 1])
        X = np.array(fileData[:, 2:], dtype=float)
        filenames = np.array(fileData[:, 0])
        
        return X, Y, featureNames, filenames
    
    def extractData(self, filePath):
        print("Extracting data from file:", filePath)

        # Extract the data
        with open(filePath, 'r') as f:
            track_data = json.load(f)
        
        # How many tracks are there?
        print(f"\tn_tracks = {len(track_data.keys())}")
        
        # What do the track Unique IDs (UIDs) look like?
        track_uids = list(track_data.keys())
        print(f"\t5 Example Track IDs = {track_uids[:5]}")
        
        # What fields are avaiable for each track?
        example_uid = track_uids[0]
        print(f"\tPer-track keys = {track_data[example_uid].keys()}")
        
        # What do the (t, x, y) track coordinates look like?
        example_coords = track_data[track_uids[0]]['txy']
        example_coords = np.array(example_coords)
        np.set_printoptions(threshold=10)
        print(f"\tCoordinate array = \n{example_coords}")
        
        # What does the label look like?
        example_label = track_data[track_uids[0]]['label']
        print(f"\tLabel = {example_label}")
        
        return track_data