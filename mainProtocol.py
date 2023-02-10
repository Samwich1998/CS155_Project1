#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:29:27 2023

@author: samuelsolomon
"""


# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General modules
import sys
import numpy as np
# Machine learning modules
import sklearn
from sklearn.linear_model import LogisticRegression
# Plotting modules
from matplotlib import pyplot as plt

# Import Data Aquisition and Analysis Files
sys.path.append('./Helper Files/')  # Folder with Data Aquisition Files
import excelProcessing       # Functions to Save/Read in Data from Excel
import generalAnalysis       # General functions

# Import Files for Machine Learning
sys.path.append('./Helper Files/Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning

if __name__ == "__main__":
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Extract the data -------------------------- #
    
    # Specify input files
    trainingFile = "./Data/train_features_20230210_060608.csv"
    unlabeledFile = "./Data/test_features_20230210_060934.csv"
    # Load in the training/testing information
    allFeatures, allLabels, featureNames, allFilenames = excelProcessing.processFiles().extractData(trainingFile)
    unlabeledFeatures, _, unlabeledFeatureNames, unlabeledFilenames = excelProcessing.processFiles().extractData(unlabeledFile)
    # Assert validity of the data
    assert all(featureNames == unlabeledFeatureNames)
    assert len(allFeatures[0]) == len(unlabeledFeatures[0]) == len(featureNames)
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Process the data -------------------------- #

    # Remove bad values
    allFeatures, allLabels, allFilenames = generalAnalysis.removeBadValues(allFeatures, allLabels, allFilenames)
    # Format
    allFilenames = np.array([filename.split("_")[0] for filename in allFilenames])
    allLabels = np.array(allLabels, dtype=int)
    
    # Standardize data
    standardizeX_Class = generalAnalysis.standardizeData(allFeatures)
    standardizedFeatures = standardizeX_Class.standardize(allFeatures)
    
    # Seperate out the lab data
    allLabels_Lab = allLabels[allFilenames == "lab"]
    standardizedFeatures_Lab = standardizedFeatures[allFilenames == "lab"]
    # Seperate out the simulated data
    allLabels_Sim = allLabels[allFilenames == "sim"][0:1000]
    standardizedFeatures_Sim = standardizedFeatures[allFilenames == "sim"][0:1000]
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Split the data --------------------------- #
    
    # Split into testing/training
    trainingFeatures, testingFeatures, trainingLabels, testingLabels = sklearn.model_selection.train_test_split(standardizedFeatures_Lab, allLabels_Lab, test_size=0.3, random_state=1, stratify=allLabels_Lab)
    
    # Add simulated data back to training
    trainingLabels = np.concatenate((trainingLabels, allLabels_Sim), axis=0)
    trainingFeatures = np.concatenate((trainingFeatures, standardizedFeatures_Sim), axis=0)
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Machine Learning -------------------------- #

    # Pick the Machine Learning Module to Use
    modelType = "RF"  # Machine Learning Options: NN, RF, LR, KNN, SVM, RG, EN, SVR
    supportVectorKernel = "linear"
    modelPath = "./Helper Files/Machine Learning/Models/predictionModel_NN1.pkl" # Path to Model (Creates New if it Doesn't Exist)
    # Choos the Folder to Save ML Results
    saveDataFolder_ML = "./Data/Machine Learning Analysis/" + modelType + "/"
        
    # Get the Machine Learning Module
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = [0, 1], saveDataFolder = saveDataFolder_ML, supportVectorKernel = supportVectorKernel)
    predictionModel = performMachineLearning.predictionModel
    
    # Predict the labels
    testScore = predictionModel.trainModel(trainingFeatures, trainingLabels, testingFeatures, testingLabels)
    predictedLabels = np.round(predictionModel.predictData(standardizedFeatures).ravel())
    predictedTestLabels = np.round(predictionModel.predictData(testingFeatures).ravel())
    finalScore = sum(predictedLabels == allLabels)/len(standardizedFeatures)
    # Calculate F2    
    f2_Score_Test = generalAnalysis.f2Score(testingLabels, predictedTestLabels)
    f2_Score_All = generalAnalysis.f2Score(allLabels, predictedLabels)
    # Print out results
    print("Test Score:", testScore)
    print("Final Score:", finalScore)
    print("Test F2 Score:", f2_Score_Test)
    print("Final F2 Score:", f2_Score_All)
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Save the Results -------------------------- #
    
    import csv     
    # name of csv file
    filename = "data/submission_RF_9Feb.csv"
    
    standardizedUnlabeledFeatures = standardizeX_Class.standardize(unlabeledFeatures)
    predictedLabels = np.round(predictionModel.predictData(standardizedUnlabeledFeatures).ravel())

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
         
        # writing the fields
        csvwriter.writerow(["uid", "label"])
         
        # writing the data rows
        csvwriter.writerows(list(zip(unlabeledFilenames, predictedLabels)))
        
        
    

            
        
        