# import random

import random 

# Function split_training_testing()

def split_training_testing(ls_data, p_test):
    """Takes a list of data and a percentage and returns a random split into a training set and a test set"""
    # Assertions
    assert type(ls_data) == list, "The data is not given as a list"
    assert type(p_test) == int, "The p_test is not an integer"
    assert 0 <= p_test <= 100, "Percent is not in range 0 to 100"

    # Testing set size 
    testing_size = int(len(ls_data) * (p_test/100))

    # Generating a random testing set 
    testing_set = random.sample(ls_data, testing_size)

    # Generating the training set 
    training_set = ls_data.copy()
    for i in testing_set:
        training_set.remove(i)

    return training_set, testing_set 

# Function confusion_matrix

def confusion_matrix(pred_values, act_values, val):
    """Takes lists of predicted and actual values and a value for the true positive class and returns TP, FP, TN, and FN (in that order) for the specified positive class"""
    # Assertions
    assert type(pred_values) == list, "Predicted values are not a list"
    assert type(act_values) == list, "Actual Values are not a list"
    assert len(pred_values) == len(act_values), "pred_values and act_values must have the same length"
    TP = 0 
    FP = 0 
    TN = 0
    FN = 0 
    for pred, actual in zip(pred_values, act_values):
        if (pred == val) and (actual == val):
            TP += 1
        elif (pred == val) and (actual != val):
            FP += 1
        elif (pred != val) and (actual != val):
            TN += 1
        else:
            FN += 1
    return TP, FP, TN, FN

# Evaluation metrics 
## accuracy function

def accuracy(TP, FP, TN, FN):
    """Takes integers or floats for TP, FP, TN, FN and returns the classification accuracy"""
    # Assertions
    assert isinstance(TP, (int, float)), "TP is not of type int or float"
    assert isinstance(FP, (int, float)), "FP is not of type int or float"
    assert isinstance(TN, (int, float)), "TN is not of type int or float"
    assert isinstance(FN, (int,float)), "FN is not of type int or float"

    try:
        return (TP + TN) / (TP + FP + TN + FN)
    except ZeroDivisionError:
        return float('nan')

## sensitivity function 
def sensitivity(TP, FN):
    """Takes integers or floats for TP and FN and returns the sensitivity or true positive rate."""
    assert isinstance(TP, (int, float)), "TP is not of type int or float"
    assert isinstance(FN, (int, float)), "FN is not of type int or float"

    try:
        return TP/ (TP + FN)
    except ZeroDivisionError:
        return float('nan') 

## specificity function 
def specificity(FP, TN):
    """Takes integers or floats for FP and TN and returns the specificity or true negative rate."""
    assert isinstance(FP, (int, float)), "FP is not of type int or float"
    assert isinstance(TN, (int, float)), "TN is not of type int or float"

    try:
        return TN / (TN + FP)
    except ZeroDivisionError:
        return float('nan')

## positive predictive value function 
def pos_pred_val(TP, FP):
    """Takes integers or floats for TP and FP and returns  the positive predictive value"""
    assert isinstance(TP, (int,float)), "TP is not of type of int or float"
    assert isinstance(FP, (int, float)), "FP is not of type int or float"

    try:
        return TP/ (TP + FP)
    except ZeroDivisionError:
        return float('nan')

## negative predictive value function
def neg_pred_val(TN, FN):
    """Takes integers or floats for TN and FN and returns the negative predicitive value"""
    assert isinstance(TN, (int, float)), "TN is not of type int or float"
    assert isinstance(FN, (int, float)), "FN is not of type int or float"

    try:
        return TN/ (TN + FN)
    except ZeroDivisionError:
        return float('nan')

# Print evaluation metrics function
def print_eval_metrics(pred_values, act_values, val):
    """Takes a list of predicted and actual values and a value for the true positive class and prints all evaluation metrics."""
    TP, FP, TN, FN = confusion_matrix(pred_values, act_values, val)
    acc = accuracy(TP, FP, TN, FN)
    sense = sensitivity(TP, FN)
    specifi = specificity(FP, TN)
    positive = pos_pred_val(TP, FP)
    negative = neg_pred_val(TN, FN)
    print(f"Accuracy: {acc}\n"
          f"Sensitivity: {sense}\n"
          f"Specificity: {specifi}\n"
          f"Positive predictive value: {positive}\n"
          f"Negative predictive value: {negative}")
    
if __name__ == '__main__':
    pass