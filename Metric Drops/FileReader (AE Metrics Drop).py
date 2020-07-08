# Reads the files which hold the performance metric percentages.
# For thresholds between 1% and 50%.
# Autoencoder model seems to stop 'running' after the 24th iteration,
# this is why the file reading method was done.
#
# William King
# 950178

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

accuracy = []
precision = []
recall = []
f1 = []
contaminations = []

# Reads each file to retrieve the respective threshold values.
for i in range(1, 51):

  # MAKE SURE YOU'RE READING FROM THE RIGHT FOLDER...
  f = open("AE Metrics Drop (Warranty Claims)\Threshold " + str(i) + ".txt", "r")
  f.readline()

  # Slices are specific to the files.
  acc = float(f.readline()[10:])
  prec = float(f.readline()[11:])
  rec = float(f.readline()[8:])
  f1Score = float(f.readline()[10:])

  accuracy.append(acc)
  precision.append(prec)
  recall.append(rec)
  f1.append(f1Score)
  contaminations.append(i)

  f.close()
  print(i)

# Plot the metrics drop for all thresholds.
plt.figure()
plt.plot(contaminations, accuracy, marker = 'o', color = 'red', label = 'Accuracy')
plt.plot(contaminations, precision, marker = 'o', color = 'blue', label = 'Precision')
plt.plot(contaminations, recall, marker = 'o', color = 'green', label = 'Recall')
plt.plot(contaminations, f1, marker = 'o', color = 'black', label = 'F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score Percentage (%)')
plt.legend()
plt.show()
 
