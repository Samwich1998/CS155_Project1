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
    
    numSim = 1000;
    # Throw away bad values
    for uid in list(rawTrainingData.keys()):
        coords = np.array(rawTrainingData[uid]['txy'])
        t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
    
        # if len(coords) < 10:
        #     rawTrainingData.pop(uid)
        # elif len(np.unique(np.diff(t))) != 1:
        #     rawTrainingData.pop(uid)
        
        if 'sim' in uid:
            numSim -= 1
        
        if numSim == 0:
            rawTrainingData.pop(uid)
        
        
    
    # sys.exit()
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Process the data -------------------------- #
        
    # Compile and save the final features
    featureExtractionFunctions.compileFeatures(rawTrainingData, 'train')
    featureExtractionFunctions.compileFeatures(rawUnlabeledData, 'test')