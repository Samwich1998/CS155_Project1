#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import os
import sys
import math
import time
import bisect
import itertools
import numpy as np
import collections
from scipy import stats
# Modules for Plotting
import matplotlib.pyplot as plt
# Neural Network Modules
from sklearn.model_selection import train_test_split

# Import Machine Learning Files
sys.path.append('./Helper Files/Machine Learning/Classification Methods/') # Folder with Machine Learning Files
import supportVectorRegression as SVR   # Functions for Support Vector Regression Algorithm
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import logisticRegression as LR         # Functions for Linear Regression Algorithm
import ridgeRegression as Ridge         # Functions for Ridge Regression Algorithm
import elasticNet as elasticNet         # Functions for Elastic Net Algorithm
import adaBoost                         # Functions for the Adaboost Algorithm
import randomForest                     # Functions for the Random Forest Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm
import xgBoost

# Import Data Extraction Files (And Their Location)
sys.path.append('./Helper Files/')  
import excelProcessing
import generalAnalysis       # General functions

# Standardize data class
class standardizeData:
    def __init__(self, X):
        self.mu_ = np.mean(X, axis=0)
        self.sigma_ = np.std(X, ddof=1, axis=0)
        
    def standardize(self, X):
        return (X - self.mu_)/self.sigma_
    
    def unStandardize(self, Xhat):
        return self.mu_ + self.sigma_*Xhat

class predictionModelHead:
    
    def __init__(self, modelType, modelPath, numFeatures, machineLearningClasses, saveDataFolder, supportVectorKernel = ""):
        # Store Parameters
        self.modelType = modelType
        self.modelPath = modelPath
        self.saveDataFolder = saveDataFolder
        self.machineLearningClasses = machineLearningClasses
        self.numClasses = len(machineLearningClasses)
        self.testSize = 0.4
        self.supportVectorKernel = supportVectorKernel
        
        self.possibleModels = ['NN', 'RF', 'LR', 'KNN', 'SVM', 'RG', 'EN', "SVR", "ADA", "XGB"]
        if modelType not in self.possibleModels:
            exit("The Model Type is Not Found")
        
        self.resetModel(numFeatures)
        if saveDataFolder:
            # Create Output File Directory to Save Data: If None
            os.makedirs(self.saveDataFolder, exist_ok=True)
    
    def resetModel(self, numFeatures = 1):
        # Holder Variables
        self.map2D = []
        # Get Prediction Model
        self.predictionModel = self.getModel(self.modelType, self.modelPath, numFeatures)        
    
    def getModel(self, modelType, modelPath, numFeatures):
        # Get the Machine Learning Model
        if modelType == "NN":
            # numFeatures = The dimensionality of one data point
            predictionModel = NeuralNet.Neural_Network(modelPath = modelPath, numFeatures = numFeatures)
        elif modelType == "RF":
            predictionModel = randomForest.randomForest(modelPath = modelPath)
        elif modelType == "ADA":
            predictionModel = adaBoost.adaBoost(modelPath = modelPath)
        elif modelType == "XGB":
            predictionModel = xgBoost.xgBoost(modelPath = modelPath)
        elif modelType == "LR":
            predictionModel = LR.logisticRegression(modelPath = modelPath)
        elif modelType == "RG":
            predictionModel = Ridge.ridgeRegression(modelPath = modelPath)
        elif modelType == "EN":
            predictionModel = elasticNet.elasticNet(modelPath = modelPath)
        elif modelType == "KNN":
            predictionModel = KNN.KNN(modelPath = modelPath, numClasses = self.numClasses)
        elif modelType == "SVM":
            predictionModel = SVM.SVM(modelPath = modelPath, modelType = self.supportVectorKernel, polynomialDegree = 3)
            # Section off SVM Data Analysis Into the Type of Kernels
            if self.saveDataFolder and self.supportVectorKernel not in self.saveDataFolder:
                self.saveDataFolder += self.supportVectorKernel +"/"
                os.makedirs(self.saveDataFolder, exist_ok=True)
        elif modelType == "SVR":
            predictionModel = SVR.supportVectorRegression(modelPath = modelPath, modelType = self.supportVectorKernel, polynomialDegree = 3)
            # Section off SVM Data Analysis Into the Type of Kernels
            if self.saveDataFolder and self.supportVectorKernel not in self.saveDataFolder:
                self.saveDataFolder += self.supportVectorKernel +"/"
                os.makedirs(self.saveDataFolder, exist_ok=True)
        else:
            print("No Matching Machine Learning Model was Found for '", modelType, "'");
            sys.exit()
        # Return the Precition Model
        return predictionModel
    
    def scoreClassificationModel(self, featureData, featureLabels, stratifyBy = [], testSplitRatio = 0.4):
        # Extract a list of unique labels
        # possibleClassifications = list(set(featureLabels))
        classificationScores = []
        # Taking the Average Score Each Time
        for _ in range(200):
            # Train the Model with the Training Data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True, stratify=stratifyBy)
            self.predictionModel.model.fit(Training_Data, Training_Labels)
            
            classAccuracies = []
            # if not testStressScores:
            #     for classification in possibleClassifications:
            #         testClassData = Testing_Data[Testing_Labels == classification]
            #         testClassLabels = self.predictionModel.model.predict(testClassData)
                    
            #         classAccuracy = len(testClassLabels[testClassLabels == classification])/len(testClassLabels)
            #         classAccuracies.append(classAccuracy)
            testClassLabels = self.predictionModel.model.predict(Testing_Data)
            classAccuracy = len(testClassLabels[testClassLabels == Testing_Labels])/len(testClassLabels)
            classAccuracies.append(classAccuracy)
            
            classificationScores.append(classAccuracies)
        
        averageClassAccuracy = stats.trim_mean(classificationScores, 0.4)
        return averageClassAccuracy
        
        
    def trainModel(self, featureData, featureLabels, featureNames = [], returnScore = False, stratifyBy = [], testSplitRatio = 0.4):
        if len(featureNames) != 0 and not len(featureNames) == len(featureData[0]):
            print("The Number of Feature Labels Provided Does Not Match the Number of Features")
            print("Removing Feature Labels")
            featureLabels = []
            
        featureData = np.array(featureData); featureLabels = np.array(featureLabels); featureNames = np.array(featureNames)
        # Find the Data Distribution
        #classDistribution = collections.Counter(featureLabels)
        # print("Class Distribution:", classDistribution)
        # print("Number of Data Points = ", len(classDistribution))
        
        if self.modelType in self.possibleModels:
            # Train the Model Multiple Times
            modelScores = []
            # Taking the Average Score Each Time
            for _ in range(300):
                # Train the Model with the Training Data
                Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True)
                modelScores.append(self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels))
            if returnScore:
                #print("Mean Testing Accuracy (Return):", meanScore)
                return stats.trim_mean(modelScores, 0.4)
            # Display the Spread of Scores
            plt.hist(modelScores, 100, facecolor='blue', alpha=0.5)
            # Fit the Mean Distribution and Save the Mean
            ae, loce, scalee = stats.skewnorm.fit(modelScores)
            # Take the Median Score as the True Score
            meanScore = np.round(loce*100, 2)
            if returnScore:
                #print("Mean Testing Accuracy (Return):", meanScore)
                return meanScore
            #print("Mean Testing Accuracy:", meanScore)
            # Label Accuracy
            self.accuracyDistributionPlot_Average(featureData, featureLabels, self.machineLearningClasses, "Test")
            self.accuracyDistributionPlot_Average(featureData, featureLabels, self.machineLearningClasses, "Full")
            # Extract Feature Importance
            #self.featureImportance(featureData, featureLabels, featureData, featureLabels, featureNames = featureNames, numTrials = 100)
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
        if self.modelType == "NN":
            # Plot the training loss    
            self.predictionModel.plotStats()
            
            
    def analyzeFeatureCombinations(self, trainingFeatures, trainingLabels, testingFeatures, testingLabels, featureNames, numFeaturesCombine, saveData = True, 
                                   saveExcelName = "Feature Combination Accuracy.xlsx", printUpdateAfterTrial = 15000, scaleLabels = True):
        # Format the incoming data
        featureNames = np.array(featureNames)
        trainingFeatures = np.array(trainingFeatures.copy())
        testingFeatures = np.array(testingFeatures.copy())
        testingLabels = np.array(testingLabels.copy())
        trainingLabels = np.array(trainingLabels.copy())
        # Function variables
        numModelsTrack = 1000   # The number of models to track
        
        # Get All Possible itertools.combinations
        self.finalPerformances = []; self.finalPerformancesF2 = []; self.featureNamesCombinations = []
        featureCombinationInds = itertools.combinations(range(0, len(featureNames)), numFeaturesCombine)
        # Find total combinations
        numFeatureCombnatons = math.comb(len(featureNames), numFeaturesCombine)
        
        t1 = time.time(); combinationRoundInd = -1
        # For Each Combination of Features
        for combinationInds in featureCombinationInds:
            combinationInds = list(combinationInds)
            combinationRoundInd += 1
            
            # Collect the Signal Data for the Specific Features
            trainingFeatures_culledFeatures = trainingFeatures[:, combinationInds]
            testingFeatures_culledFeatures = testingFeatures[:, combinationInds]
            # Collect the Specific Feature Names
            featureNamesCombination_String = ''
            for name in featureNames[combinationInds]:
                featureNamesCombination_String += name + ' '
                
            # print(featureNamesCombination_String)
            
            # Reset the Input Variab;es
            modelPerformances = []
            self.resetModel() # Reset the ML Model

            # Score the model with this data set.
            modelPerformances.append(self.predictionModel.trainModel(trainingFeatures_culledFeatures, trainingLabels, testingFeatures_culledFeatures, testingLabels))
            
            # if numFeaturesCombine != 1:
            # self.findWorstSubject(trainingFeatures_culledFeatures, featureLabels, subjectOrder, featureNamesCombination_String)
                    
            modelPerformance = stats.trim_mean(modelPerformances, 0.3)
            # If the model's performance is one of the top scores
            if len(self.finalPerformances) < numModelsTrack or modelPerformance > self.finalPerformances[-1]:
                predictedTestLabels = np.round(self.predictionModel.predictData(testingFeatures_culledFeatures).ravel())
                f2_Score_Test = generalAnalysis.f2Score(testingLabels, predictedTestLabels)
                
                insertionPoint = bisect.bisect(self.finalPerformances, -modelPerformance, key=lambda x: -x)
                # Save the model score and standard deviation
                self.featureNamesCombinations.insert(insertionPoint, featureNamesCombination_String[0:-1])
                self.finalPerformances.insert(insertionPoint, modelPerformance)
                self.finalPerformancesF2.insert(insertionPoint, f2_Score_Test)
                
                # Only track the best models
                if len(self.finalPerformances) > numModelsTrack:
                    self.finalPerformances.pop()
                    self.finalPerformancesF2.pop()
                    self.featureNamesCombinations.pop()
                
            # Report an Update Every Now and Then
            if (combinationRoundInd%printUpdateAfterTrial == 0 and combinationRoundInd != 0) or combinationRoundInd == 20:
                t2 = time.time()
                percentComplete = 100*combinationRoundInd/numFeatureCombnatons
                setionPercent = 100*min(combinationRoundInd or 1, printUpdateAfterTrial)/numFeatureCombnatons
                print(str(np.round(percentComplete, 2)) + "% Complete; Estimated Time Remaining: " + str(np.round((t2-t1)*(100-percentComplete)/(setionPercent*60), 2)) + " Minutes")
                t1 = time.time()
                    
        print("\t", self.finalPerformances[0], self.finalPerformancesF2[0], self.featureNamesCombinations[0])
        # Save the Data in Excel
        if saveData:
            excelProcessing.processMLData().saveFeatureComparison(np.dstack((self.finalPerformances, self.finalPerformancesF2, self.featureNamesCombinations))[0], [], ["Mean Score", "STD", "Feature Combination"], self.saveDataFolder, saveExcelName, sheetName = str(numFeaturesCombine) + " Features in Combination", saveFirstSheet = True)
        return np.array(self.finalPerformances), np.array(self.finalPerformancesF2), np.array(self.featureNamesCombinations)
    
    def getSpecificFeatures(self, allFeatureNames, getFeatureNames, featureData):
        featureData = np.array(featureData)
        
        newfeatureData = []
        for featureName in getFeatureNames:
            featureInd = list(allFeatureNames).index(featureName)
            
            if len(newfeatureData) == 0:
                newfeatureData = featureData[:,featureInd]
            else:
                newfeatureData = np.dstack((newfeatureData, featureData[:,featureInd]))
        
        if len(newfeatureData) == 0:
            print("No Features grouped")
            return []
        return newfeatureData[0]

    def countScoredFeatures(self, featureCombinations):
        allFeatureAppearance = []
        # Create list of all features that appear in the vombinations
        for featureCombination in featureCombinations:
            allFeatureAppearance.extend(featureCombination.split(" "))
        
        # Count each feature in the list and return the counter
        bestFeaturesCounter = collections.Counter(allFeatureAppearance)
        featureFound, featureFoundCounter = zip(*bestFeaturesCounter.items())
        return  np.array(featureFound), np.array(featureFoundCounter)
    
    
            
        