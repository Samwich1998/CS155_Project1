#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Need to Install on the Anaconda Prompt:
    $ pip install pyexcel
"""


# Basic Modules
import os
import sys
import numpy as np


# Standardize data class
class standardizeData:
    def __init__(self, X):
        self.mu_ = np.mean(X, axis=0)
        self.sigma_ = np.std(X, ddof=1, axis=0)
        
    def standardize(self, X):
        return (X - self.mu_)/self.sigma_
    
    def unStandardize(self, Xhat):
        return self.mu_ + self.sigma_*Xhat
    

def removeBadValues(X, Y, filenames):
    # Find values to remove
    removeMask = np.any(np.isnan(X), axis=1)
    # Remove the values
    X = X[~removeMask]
    Y = Y[~removeMask]
    filenames = filenames[~removeMask]
    
    return X, Y, filenames



def f2Score(y_test, y_pred_class):
    
    # Calculate the true positive count
    tp = np.sum((y_test == 1) & (y_pred_class == 1))
    
    # Calculate the false positive count
    fp = np.sum((y_test == 0) & (y_pred_class == 1))
    
    # Calculate the false negative count
    fn = np.sum((y_test == 1) & (y_pred_class == 0))
    
    # Calculate precision
    precision = tp / (tp + fp)
    
    # Calculate recall
    recall = tp / (tp + fn)
    
    # Calculate the F2 score
    f2_score = 5 * (precision * recall) / ((4 * precision) + recall)
    
    return f2_score