# Importing functions 
import math 
import unittest 
import class_eval

class Test_split(unittest.TestCase):
    """Tests for split_training_testing function"""
    def test_split_zero_percent(self):
        """Takes 0% and returns full training set and empty test set"""
        data = [1, 2, 3, 4, 5, 6]
        train, test = class_eval.split_training_testing(data, 0)
        output_train = data 
        output_test = []
        self.assertEqual(train, output_train)
        self.assertEqual(test, output_test)

    def test_split_hundred(self):
        """Takes 100% and returns empty training set and a full test set"""
        data = [1, 2, 3, 4, 5, 6]
        train, test = class_eval.split_training_testing(data, 100)
        output_train = []
        output_test = [1, 2, 3, 4, 5, 6]
        self.assertEqual(train, output_train)
        self.assertEqual(sorted(test), output_test)

    def test_split_middle(self):
        """Takes a mid-range percent and splits data proportionally."""
        data = list(range(10))
        train, test = class_eval.split_training_testing(data, 40)
        output_train_len = 6 
        output_test_len = 4 
        self.assertEqual(len(train), output_train_len)
        self.assertEqual(len(test), output_test_len)
        self.assertEqual(sorted(train + test), data)

    def test_split_with_duplicates(self):
        """Takes a list with duplicates and correctly computes set sizes"""
        data = [1, 1, 2, 2, 3, 3]
        train, test = class_eval.split_training_testing(data, 50)
        output_train_len = 3 
        output_test_len = 3 
        total_len = len(data)
        self.assertEqual(len(train), output_train_len)
        self.assertEqual(len(test), output_test_len)
        self.assertEqual(len(train) + len(test), total_len)

# Tests for Confusion Matrix

class TestConfusionMatrix(unittest.TestCase):
    """Tests for the confusion_matrix() function"""

    def test_confusion_perfectmatch(self):
        """Takes identical lists and returns all TPs"""
        pred = [1, 1, 1]
        act = [1, 1, 1]
        TP, FP, TN, FN = class_eval.confusion_matrix(pred, act, 1)
        output_TP = 3 
        output_FP = 0 
        output_TN = 0 
        output_FN = 0 
        self.assertEqual(TP, output_TP)
        self.assertEqual(FP, output_FP)
        self.assertEqual(TN, output_TN)
        self.assertEqual(FN, output_FN)
    
    def test_confusion_mismatch(self):
        """Takes opposite lists and returns all FPs"""
        pred = [1,1,1]
        act = [0,0,0]
        TP, FP, TN, FN = class_eval.confusion_matrix(pred, act, 1)
        output_TP = 0 
        output_FP = 3 
        output_TN = 0 
        output_FN = 0 
        self.assertEqual(TP, output_TP)
        self.assertEqual(FP, output_FP)
        self.assertEqual(TN, output_TN)
        self.assertEqual(FN, output_FN)

    def test_confusion_mixed(self):
        """Takes a mixed list and returns 1 TP, 1 FP, 1 TN, 1 FN"""
        pred = [1,1,0,0]
        act= [1,0,1,0]
        TP, FP, TN, FN = class_eval.confusion_matrix(pred, act, 1)
        output_TP = 1 
        output_FP = 1 
        output_TN = 1 
        output_FN = 1
        self.assertEqual(TP, output_TP)
        self.assertEqual(FP, output_FP)
        self.assertEqual(TN, output_TN)
        self.assertEqual(FN, output_FN)

    def test_confusion_nopositive(self):
        """Returns no TPs or FPs, only TNs and FNs"""
        pred = [0,0,0]
        act = [1,0,1]
        TP, FP, TN, FN = class_eval.confusion_matrix(pred,act,1)
        output_TP = 0 
        output_FP = 0
        output_TN = 1 
        output_FN = 2
        self.assertEqual(TP, output_TP)
        self.assertEqual(FP, output_FP)
        self.assertEqual(TN, output_TN)
        self.assertEqual(FN, output_FN)

# Testing Accuracy 
class TestAccuracy(unittest.TestCase):
    """Tests for the accuracy function"""

    def test_accuracy_normal(self):
        """Standard accuracy case"""
        TP = 4
        FP = 1 
        TN = 3 
        FN = 2 
        output = class_eval.accuracy(TP, FP, TN, FN)
        expected_output = 0.7
        self.assertEqual(output, expected_output)
    
    def test_accuracy_zero_numerator(self):
        """TP + TN = 0"""
        TP = 0
        FP = 3 
        TN = 0
        FN = 2 
        output = class_eval.accuracy(TP, FP, TN, FN)
        expected_output = 0
        self.assertEqual(output, expected_output)

    def test_accuracy_zero_denominator(self):
        """Returns NaN when denominator is 0"""
        TP = 0
        FP = 0 
        TN = 0
        FN = 0 
        output = class_eval.accuracy(TP, FP, TN, FN)
        self.assertTrue(math.isnan(output))

# Test for Sensitivity 
class TestSensitivitiy(unittest.TestCase):
    """Tests for Sensitivity function"""
    def test_sensitivity_normal(self):
        TP = 6
        FN = 4 
        output = class_eval.sensitivity(TP, FN)
        expected_output = 0.6
        self.assertEqual(output, expected_output)

    def test_sensitivity_zero_numerator(self):
        """TP = 0"""
        TP = 0 
        FN = 3
        output = class_eval.sensitivity(TP, FN)
        expected_output = 0 
        self.assertEqual(output, expected_output)
    
    def test_sensitivity_zero_denominator(self):
        """Zero denominator gives Nan"""
        TP = 0 
        FN = 0 
        output = class_eval.sensitivity(TP, FN)
        self.assertTrue(math.isnan(output))

# Test Specificity 
class TestSpecificity(unittest.TestCase):
    """Tests for Specificity"""

    def test_specificity_normal(self):
        """Standard case"""
        FP = 4
        TN = 6 
        output = class_eval.specificity(FP, TN)
        expected_output = 0.6 
        self.assertEqual(output, expected_output)

    def test_specificity_zero_numerator(self):
        """Numerator is 0"""
        FP = 4
        TN = 0
        output = class_eval.specificity(FP, TN)
        expected_output = 0 
        self.assertEqual(output, expected_output)

    def test_specificity_zero_denominator(self):
        """Numerator is 0"""
        FP = 0
        TN = 0
        output = class_eval.specificity(FP, TN)
        self.assertTrue(math.isnan(output))
    
class TestPositivePredictiveValue(unittest.TestCase):
    """Tests for pos_pred_val"""
    def test_ppv_normal(self):
        """Standard ppv case"""
        TP = 5
        FP = 5 
        output = class_eval.pos_pred_val(TP, FP)
        expected_output = 0.5 
        self.assertEqual(output, expected_output)

    def test_ppv_zero_numerator(self):
        """TP = 0"""
        TP = 0
        FP = 5 
        output = class_eval.pos_pred_val(TP, FP)
        expected_output = 0
        self.assertEqual(output, expected_output)
    
    def test_ppv_zero_denominator(self):
        """TP = FP = 0"""
        TP = 0
        FP = 0 
        output = class_eval.pos_pred_val(TP, FP)
        self.assertTrue(math.isnan(output))
    
class TestNegativePredictiveValue(unittest.TestCase):
    """Tests for neg_pred_val"""

    def test_npv_normal(self):
        """Standard NPV case"""
        TN = 6 
        FN = 4 
        output = class_eval.neg_pred_val(TN, FN)
        expected_output = 0.6 
        self.assertEqual(output, expected_output)

    def test_npv_zero_numerator(self):
        """TN = 0"""
        TN = 0 
        FN = 4
        output = class_eval.neg_pred_val(TN, FN)
        expected_output = 0 
        self.assertEqual(output, expected_output)

    def test_npv_zero_denominator(self):
        """Zero denominator returns NaN"""
        TN = 0 
        FN = 0
        output = class_eval.neg_pred_val(TN, FN)
        self.assertTrue(math.isnan(output))

if __name__ == '__main__':
    unittest.main()