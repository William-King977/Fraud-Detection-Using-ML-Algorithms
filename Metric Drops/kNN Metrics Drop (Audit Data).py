# Creates plots of kNN running on Audit Data
# with thresholds between 1% and 50%.
# For each threshold, kNN will run with the value k between 1 and 20.
# Each plot will represent kNN running with a single threshold, but
# with k changing from 1 to 20.
#
# William King
# 950178

import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.knn import KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Import the data set.
audit_df = pd.read_csv('Datasets/audit_data/audit_risk.csv')
anomaly_points_df = audit_df['Risk']
anomaly_points = anomaly_points_df.values.reshape(-1, 1)

# Fill NaN values with 0.
audit_df.fillna(0, inplace = True)

# Data Preparation.
del audit_df['LOCATION_ID']
del audit_df['TOTAL']
del audit_df['Risk']
standardised_data = StandardScaler().fit_transform(audit_df)
samples = standardised_data
k_set = [x for x in range(1, 21)]
 
# Runs kNN with thresholds between 1% and 50%.
for i in range(1, 51):
  accuracy = []
  precision = []
  recall = []
  f1 = []
  contaminations = []
  outliers = float(i) / 100.0

  # Runs kNN 20 times with the same threshold, but with k between 1 and 20.
  for k in k_set:
    model = KNN(n_neighbors = k, contamination = outliers)
    model.fit(samples)
    pred = model.predict(samples)

    # # Stores the output of performance metrics.
    accuracy.append(accuracy_score(anomaly_points, pred) * 100)
    precision.append(precision_score(anomaly_points, pred) * 100)
    recall.append(recall_score(anomaly_points, pred) * 100)
    f1.append(f1_score(anomaly_points, pred) * 100)
    contaminations.append(i)

  # Create the plot and save it.
  plt.figure()
  plt.plot(k_set, accuracy, marker = 'o', color = 'red', label = 'Accuracy')
  plt.plot(k_set, precision, marker = 'o', color = 'blue', label = 'Precision')
  plt.plot(k_set, recall, marker = 'o', color = 'green', label = 'Recall')
  plt.plot(k_set, f1, marker = 'o', color = 'black', label = 'F1 Score')
  plt.title("Threshold: " + str(outliers))
  plt.xlabel('k')
  plt.ylabel('Score Percentage (%)')
  plt.legend()
  plt.savefig('kNN Metrics Drop (Audit Data)/kNN ' + str(outliers) + '.png')

