#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Need to Install on the Anaconda Prompt:
    $ pip install pyexcel
"""


# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General modules
import sys
import numpy as np

# Import Data Aquisition and Analysis Files
sys.path.append('./Helper Files/')  # Folder with Data Aquisition Files
import excelProcessing       # Functions to Save/Read in Data from Excel
import featureExtractionFunctions

if __name__ == "__main__":
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Extract the data -------------------------- #
    
    # Specify input files
    rawTrainingDataFile = "./Data/Input Data/train.json"
    rawUnlabeledDataFile = "./Data/Input Data/test.json"
    # Load in the training/testing information
    rawTrainingData = excelProcessing.processFiles().extractData(rawTrainingDataFile)
    rawUnlabeledData = excelProcessing.processFiles().extractData(rawUnlabeledDataFile)
    
    xyt = []
    #numSim = 500;
    # Throw away bad values
    for uid in list(rawTrainingData.keys()):
        coords = np.array(rawTrainingData[uid]['txy'])
        t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
        
        t = np.array(t).reshape(-1,1)
        x = np.array(x).reshape(-1,1)
        y = np.array(y).reshape(-1,1)
        allCoords = np.concatenate((t, x, y), axis=1)
        xyt.append(allCoords)
    
        if len(coords) < 5:
            rawTrainingData.pop(uid)
            continue
        elif len(np.unique(np.diff(t))) != 1:
            rawTrainingData.pop(uid)
            continue
        
        # if 'sim' in uid:
        #     numSim -= 1
        # if numSim <= 0:
        #     rawTrainingData.pop(uid)
        #     continue
        
        
            
    import tensorflow as tf
    
    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(32, input_shape=(None, 3)))
    model.add(tf.keras.layers.Dense(1))
    
    # Compile the model with an optimizer and a loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    
    # Generate some dummy data for demonstration purposes
    outputs = tf.random.normal(shape=(1, 300, 1))

    # Train the model
    model.fit(xyt, outputs, epochs=10, batch_size=32)
    
    # sys.exit()
        
    
    # sys.exit()
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Process the data -------------------------- #
        
    # Compile and save the final features
    featureExtractionFunctions.compileFeatures(rawTrainingData, 'train')
    featureExtractionFunctions.compileFeatures(rawUnlabeledData, 'test')