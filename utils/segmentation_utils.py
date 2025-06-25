import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from tqdm import tqdm

# Function to apply K-means clustering
def apply_kmeans(image, k):
    # Reshape the image for clustering
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map labels to centers and reshape back to the original image shape
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Also return the labels for evaluation
    labels_reshaped = labels.reshape(image.shape[:2])
    return segmented_image, labels_reshaped


# Function to binarize the mask (assuming water/flood areas are darker)
def binarize_mask(mask):
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # return binary
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Contoh range warna gelap/coklat/air — ini bisa diatur
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 80])

    binary = cv2.inRange(hsv, lower, upper)
    return binary

# Function to evaluate segmentation results
def evaluate_segmentation(kmeans_labels, binary_mask, k):
    # For K=2, we need to determine which cluster corresponds to the flood area
    if k == 2:
        # Try both possible mappings and take the one with higher accuracy
        mapping1 = {0: 0, 1: 255}
        mapping2 = {0: 255, 1: 0}

        pred1 = np.vectorize(mapping1.get)(kmeans_labels)
        pred2 = np.vectorize(mapping2.get)(kmeans_labels)

        acc1 = np.sum(pred1 == binary_mask) / binary_mask.size
        acc2 = np.sum(pred2 == binary_mask) / binary_mask.size

        if acc1 > acc2:
            pred = pred1
        else:
            pred = pred2
    else:
        # For K>2, we need to map each cluster to either 0 or 255
        # First, let's find the average intensity in the original mask for each cluster
        mapping = {}
        for label in range(k):
            mask_values = binary_mask[kmeans_labels == label]
            if len(mask_values) > 0:
                avg_value = np.mean(mask_values)
                if avg_value > 127:
                    mapping[label] = 255
                else:
                    mapping[label] = 0

            else:
                mapping[label] = 0

        pred = np.zeros_like(kmeans_labels)
        for label, value in mapping.items():
            pred[kmeans_labels == label] = value

    # Flatten arrays for metric calculation
    y_true = binary_mask.flatten()
    y_pred = pred.flatten()

    # Convert to binary for sklearn metrics (0 and 1)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }, pred