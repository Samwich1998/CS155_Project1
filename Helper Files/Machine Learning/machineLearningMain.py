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
import pandas as pd
from scipy import stats
# Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
# Machine Learning Modules
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
# Neural Network Modules
from sklearn.model_selection import train_test_split
# Feature Importance
import shap

# Import Machine Learning Files
sys.path.append('./Helper Files/Machine Learning/Classification Methods/') # Folder with Machine Learning Files
import supportVectorRegression as SVR   # Functions for Support Vector Regression Algorithm
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import logisticRegression as LR         # Functions for Linear Regression Algorithm
import ridgeRegression as Ridge         # Functions for Ridge Regression Algorithm
import elasticNet as elasticNet         # Functions for Elastic Net Algorithm
import randomForest                     # Functions for the Random Forest Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm

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
        
        self.possibleModels = ['NN', 'RF', 'LR', 'KNN', 'SVM', 'RG', 'EN', "SVR"]
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
            
    
    def findWorstSubject(self, featureData, featureLabels, subjectOrder, featureNamesCombination_String):
        subjectOrder = np.array(subjectOrder)
        featureLabels = np.array(featureLabels)
        
        # Get All Possible itertools.combinations
        subjectCombinationInds = itertools.combinations(range(0, len(featureLabels)),  len(featureLabels) - 3)
        # subjectCombinationInds = itertools.combinations(range(0, len(subjectOrder)),  len(subjectOrder) - 2)        
        allSubjectLabels = set(range(0, len(featureLabels)))
        
        removedInds = []
        removedSubjects = []
        finalPerformances = []
        for subjectInds in subjectCombinationInds:
            # subjectIndsReal = []
            # for subjectInd in subjectInds:
            #     subjectIndsReal.extend([subjectInd*6+j for j in range(6)])
            # subjectInds = subjectIndsReal
            
            # Reset the Input Variab;es
            self.resetModel() # Reset the ML Model
            
            culledSubjectData =  featureData[subjectInds, :]
            culledSubjectLabels = featureLabels[list(subjectInds)]

            # Score the model with this data set.
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = culledSubjectData, culledSubjectData, culledSubjectLabels, culledSubjectLabels
            modelPerformance = self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
            # Save which subjects were removed
            discardedSubjectInds = np.array(list(allSubjectLabels.difference(set(subjectInds))))
            removedSubject = subjectOrder[discardedSubjectInds]
        
            insertionPoint = bisect.bisect(finalPerformances, -modelPerformance, key=lambda x: -x)
            # Save the model score and standard deviation
            removedInds.insert(insertionPoint, discardedSubjectInds)
            removedSubjects.insert(insertionPoint, removedSubject)
            finalPerformances.insert(insertionPoint, modelPerformance)
            
        print(featureNamesCombination_String, finalPerformances[0], finalPerformances[-1], removedSubjects[0], removedInds[0])
        # print(finalPerformances[0], finalPerformances[-1], set(removedSubjects[0]), min(removedInds[0]), max(removedInds[0]))
                    
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
                    
        print(self.finalPerformances[0], self.finalPerformancesF2[0], self.featureNamesCombinations[0])
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
    
    
    


    def mapTo2DPlot(self, featureData, featureLabels, name = "Channel Map"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(featureData, featureLabels)
        
        mds = MDS(n_components=2,random_state=0, n_init = 4)
        X_2d = mds.fit_transform(X_scaled)
        
        X_2d = self.rotatePoints(X_2d, -np.pi/2).T
        
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = featureLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
        return X_2d
    
    def rotatePoints(self, rotatingMatrix, theta_rad = -np.pi/2):

        A = np.matrix([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad), np.cos(theta_rad)]])
        
        m2 = np.zeros(rotatingMatrix.shape)
        
        for i,v in enumerate(rotatingMatrix):
          w = A @ v.T
          m2[i] = w
        m2 = m2.T
        
        return m2
    
    
    def plot3DLabels(self, featureData, featureLabels, name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter3D(featureData[:, 3], featureData[:, 1], featureData[:, 2], c = featureLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 100, edgecolors='k')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
    
    def plot3DLabelsMovie(self, featureData, featureLabels, name = "Channel Feature Distribution Movie"):
        # Plot and Save
        fig = plt.figure()
        #fig.set_size_inches(15,15,10)
        ax = plt.axes(projection='3d')
        
        # Initialize Relevant Channel 4 Range
        errorPoint = 0.01; # Width of Channel 4's Values
        channel4Vals = np.arange(min(featureData[:, 3]), max(featureData[:, 3]), 2*errorPoint)
        
        # Initialize Movie Writer for Plots
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=name + " " + self.modelType, artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        
        with writer.saving(fig, self.saveDataFolder + name + " " + self.modelType + ".mp4", 300):
            for channel4Val in channel4Vals:
                channelPoints1 = featureData[:, 0][abs(featureData[:, 3] - channel4Val) < errorPoint]
                channelPoints2 = featureData[:, 1][abs(featureData[:, 3] - channel4Val) < errorPoint]
                channelPoints3 = featureData[:, 2][abs(featureData[:, 3] - channel4Val) < errorPoint]
                currentLabels = featureLabels[abs(featureData[:, 3] - channel4Val) < errorPoint]
                
                if len(currentLabels) != 0:
                    # Scatter Plot
                    figMap = ax.scatter3D(channelPoints1, channelPoints2, channelPoints3, "o", c = currentLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 50, edgecolors='k')
        
                    ax.set_title('Channel Feature Distribution; Channel 4 = ' + str(channel4Val) + " ± " + str(errorPoint));
                    ax.set_xlabel("Channel 1")
                    ax.set_ylabel("Channel 2")
                    ax.set_zlabel("Channel 3")
                    ax.yaxis._axinfo['label']['space_factor'] = 20
                    
                    ax.set_xlim3d(0, max(featureData[:, 0]))
                    ax.set_ylim3d(0, max(featureData[:, 1]))
                    ax.set_zlim3d(0, max(featureData[:, 2]))
                    
                    # Figure Aesthetics
                    cb = fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
                    plt.rcParams['figure.dpi'] = 300
                    figMap.set_clim(-0.5, 5.5)
                    
                    # Write to Video
                    writer.grab_frame()
                    # Clear Previous Frame
                    plt.cla()
                    cb.remove()
                
        plt.show() # Must be the Last Line
                 
    def accuracyDistributionPlot_Average(self, featureData, featureLabels, machineLearningClasses, analyzeType = "Full", name = "Accuracy Distribution", testSplitRatio = 0.4):
        numAverage = 200
        
        accMat = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
        # Taking the Average Score Each Time
        for roundInd in range(1,numAverage+1):
            # Train the Model with the Training Data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True, stratify=featureLabels)
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
            if analyzeType == "Full":
                inputData = featureData; inputLabels = featureLabels
            elif analyzeType == "Test":
                inputData = Testing_Data; inputLabels = Testing_Labels
            else:
                sys.exit("Unsure which data to use for the accuracy map");

            testingLabelsML = self.predictionModel.predictData(inputData)
            # Calculate the Accuracy Matrix
            accMat_Temp = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
            for ind, channelFeatures in enumerate(inputData):
                # Sum(Row) = # of Gestures Made with that Label
                # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
                accMat_Temp[inputLabels[ind]][testingLabelsML[ind]] += 1
        
            # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat_Temp[label] = 100*accMat_Temp[label]/(np.sum(accMat_Temp[label]))
            
                # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat[label] = (accMat[label]*(roundInd-1) + accMat_Temp[label])/roundInd

        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
                
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(5,5)
        
        # Make heatmap on plot
        im = createMap.heatmap(accMat, machineLearningClasses, machineLearningClasses, ax=ax,
                           cmap="binary")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'serif',
                'serif': 'Ubuntu',
                'size'   : 20}
        matplotlib.rc('font', **font)

        
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(self.saveDataFolder + name + " " + analyzeType + " " + self.modelType + ".png", dpi=130, bbox_inches='tight')
        plt.show()
    
    
    def plotImportance(self, perm_importance_result, featureLabels, name = "Relative Feature Importance"):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if len(featureLabels) != 0:
            _ = ax.set_yticklabels(np.array(featureLabels)[indices])
      #      headers = np.array(featureLabels)[indices]
      #      for i in headers:
      #          print('%s Weight: %.5g' % (str(i),v))
        plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
        
    
    def featureImportance(self, featureData, featureLabels, Testing_Data, Testing_Labels, featureNames = [], numTrials = 100):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
      #  if self.modelType not in ["NN"]:
      #      importanceResults = permutation_importance(self.predictionModel.model, featureData, featureLabels, n_repeats=numTrials)
      #      self.plotImportance(importanceResults, featureNames)
        
        if self.modelType == "RF":
            # get importance
            importance = self.predictionModel.model.feature_importances_
            # summarize feature importance
            for i,v in enumerate(importance):
                if len(featureNames) != 0:
                    i = featureNames[i]
                    print('%s Weight: %.5g' % (str(i),v))
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            freq_series = pd.Series(importance)
            ax = freq_series.plot(kind="bar")
            
            # Specify Figure Aesthetics
            ax.set_title("Feature Importance in Model")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Feature Importance")
            
            # Set X-Labels
            if len(featureNames) != 0:
                ax.set_xticklabels(featureNames)
                self.add_value_labels(ax)
            # Show Plot
            name = "Feature Importance"
            plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
            pyplot.show()
             
        
        if len(featureNames) != 0:
            featureNames = np.array(featureNames)
            print("Entering SHAP Analysis")
            # Make Output Folder for SHAP Values
            os.makedirs(self.saveDataFolder + "SHAP Values/", exist_ok=True)
            # Create Panda DataFrame to Match Input Type for SHAP
            testingDataPD = pd.DataFrame(featureData, columns = featureNames)
            
            # More General Explainer
            explainerGeneral = shap.Explainer(self.predictionModel.model.predict, testingDataPD)
            shap_valuesGeneral = explainerGeneral(testingDataPD)
            
            # MultiClass (Only For Tree)
            if self.modelType == "RF":
                explainer = shap.TreeExplainer(self.predictionModel.model)
                shap_values = explainer.shap_values(testingDataPD)
                
                # misclassified = Testing_Labels != self.predictionModel.model.predict(featureData)
                # shap.multioutput_decision_plot(list(explainer.expected_value), list(shap_values), row_index = 0, features = testingDataPD, feature_names = list(featureNames), feature_order = "importance", highlight = misclassified)
                # #shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = list(featureNames), feature_order = "importance", highlight = misclassified)

            else:
                # Calculate Shap Values
                explainer = shap.KernelExplainer(self.predictionModel.model.predict, testingDataPD)
                shap_values = explainer.shap_values(testingDataPD, nsamples=len(featureData))
            
            return shap_values
            
            # Specify Indivisual Sharp Parameters
            dataPoint = 3
            featurePoint = 2
            explainer.expected_value = 0
            
            
            # Summary Plot
            name = "Summary Plot"
            summaryPlot = plt.figure()
            if self.modelType == "RF":
                shap.summary_plot(shap_values, testingDataPD, plot_type="bar", class_names=self.machineLearningClasses, feature_names = featureNames, max_display = len(featureNames))
            else:
                shap.summary_plot(shap_valuesGeneral, testingDataPD, class_names=self.machineLearningClasses, feature_names = featureNames)
            summaryPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Dependance Plot
            name = "Dependance Plot"
            dependancePlot, dependanceAX = plt.subplots()
            shap.dependence_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureNames, ax = dependanceAX)
            dependancePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Indivisual Force Plot
            name = "Indivisual Force Plot"
            forcePlot = shap.force_plot(explainer.expected_value, shap_values[dataPoint,:], features = np.round(testingDataPD.iloc[dataPoint,:], 5), feature_names = featureNames, matplotlib = True, show = False)
            forcePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Full Force Plot. NOTE: CANNOT USE matplotlib = True to See
            name = "Full Force Plot"
            fullForcePlot = shap.force_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureNames, matplotlib = False, show = True)
            shap.save_html(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".htm", fullForcePlot)
            
            # WaterFall Plot
            name = "Waterfall Plot"
            waterfallPlot = plt.figure()
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0dataPoint], feature_names = featureNames, max_display = len(featureNames), show = True)
            shap.plots.waterfall(shap_valuesGeneral[dataPoint],  max_display = len(featureNames), show = True)
            waterfallPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
 
            # Indivisual Decision Plot
            misclassified = featureLabels != self.predictionModel.model.predict(featureData)
            decisionFolder = self.saveDataFolder + "SHAP Values/Decision Plots/"
            os.makedirs(decisionFolder, exist_ok=True) 
            # for dataPoint1 in range(len(testingDataPD)):
            #     name = "Indivisual Decision Plot DataPoint Num " + str(dataPoint1)
            #     decisionPlot = plt.figure()
            #     shap.decision_plot(explainer.expected_value, shap_values[dataPoint1,:], features = testingDataPD.iloc[dataPoint1,:], feature_names = featureNames, feature_order = "importance")
            #     decisionPlot.savefig(decisionFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Decision Plot
            name = "Decision Plot"
            decisionPlotOne = plt.figure()
            shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureNames, feature_order = "importance")
            decisionPlotOne.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Bar Plot
            name = "Bar Plot"
            barPlot = plt.figure()
            shap.plots.bar(shap_valuesGeneral, max_display = len(featureNames), show = True)
            barPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)

            # name = "Segmented Bar Plot"
            # barPlotSegmeneted = plt.figure()
            # labelTypesNums = [0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2]
            # labelTypes = [listOfStressors[ind] for ind in labelTypesNums]
            # shap.plots.bar(shap_valuesGeneral.cohorts(labelTypes).abs.mean(0))
            # barPlotSegmeneted.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + "_Segmented.png", bbox_inches='tight', dpi=300)

                
            # HeatMap Plot
            name = "Heatmap Plot"
            heatmapPlot = plt.figure()
            shap.plots.heatmap(shap_valuesGeneral, max_display = len(featureNames), show = True, instance_order=shap_valuesGeneral.sum(1))
            heatmapPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                
            # Scatter Plot
            scatterFolder = self.saveDataFolder + "SHAP Values/Scatter Plots/"
            os.makedirs(scatterFolder, exist_ok=True)
            for featurePoint1 in range(len(featureLabels)):
                for featurePoint2 in range(len(featureLabels)):
                    name = "Scatter Plot (" + featureNames[featurePoint1] + " VS " + featureNames[featurePoint2] + ")" 
                    scatterPlot, scatterAX = plt.subplots()
                    shap.plots.scatter(shap_valuesGeneral[:, featureLabels[featurePoint1]], color = shap_valuesGeneral[:, featureLabels[featurePoint2]], ax = scatterAX)
                    scatterPlot.savefig(scatterFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Monitoring Plot (The Function is a Beta Test As of 11-2021)
            if len(featureData) > 150:  # They Skip Every 50 Points I Believe
                name = "Monitor Plot"
                monitorPlot = plt.figure()
                shap.monitoring_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureNames)
                monitorPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                          
    def add_value_labels(self, ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.
    
        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """
    
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
    
            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'
    
            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'
    
            # Use Y value as label and format number with one decimal place
            label = "{:.3f}".format(y_value)
    
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
        
        