# Creates a plot showing Autoencoder's performances on Audit Data
# with thresholds between 1& and 50%.
# The results of each run of Autoencoder will be stored in
# a text file. The reason being that Autoencoder has long training times
# and the program tends to stop mid execution.
#
# William King
# 950178

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder

# Import the data set.
audit_df = pd.read_csv('Datasets/audit_data/audit_risk.csv')
anomaly_points_df = audit_df['Risk']
anomaly_points = anomaly_points_df.values.reshape(-1, 1)

# Fill NaN values with 0.
audit_df.fillna(0, inplace = True)

# Data preperation.
del audit_df['LOCATION_ID']
del audit_df['TOTAL']
del audit_df['Risk']
standardised_data = StandardScaler().fit_transform(audit_df)
samples = standardised_data

# Run the autoencoder model for every threshold between 1% and 50%.
for i in range(0, 51):
  accuracy = []
  precision = []
  recall = []
  f1 = []
  contaminations = []
  outliers = float(i) / 100.0
  
  model = AutoEncoder(hidden_neurons = [24, 12, 6, 2, 6, 12, 24],
                                epochs = 100,
                                verbose = 0,
                                contamination = outliers)
  model.fit(samples)
  pred = model.predict(samples)

  # Stores the output of performance metrics.
  acc = accuracy_score(anomaly_points, pred) * 100
  prec = precision_score(anomaly_points, pred) * 100
  rec = recall_score(anomaly_points, pred) * 100
  f1Score = f1_score(anomaly_points, pred) * 100
  accuracy.append(acc)
  precision.append(prec)
  recall.append(rec)
  f1.append(f1Score)
  contaminations.append(i)

  # Write down the output of each metric for the current threshold.
  # Helps keep track of values if the program 'stops' mid execution.
  f = open("AE Metrics Drop (Audit Data)\Threshold " + str(i) + ".txt", "w")
  f.write(str(i) + "\n")
  f.write("Accuracy: " + str(acc)+ " \n")
  f.write("Precision: " + str(prec)+ " \n")
  f.write("Recall: " + str(rec)+ " \n")
  f.write("F1 Score: " + str(f1Score)+ " \n")
  f.write(" \n")
  f.close()
  print(i)

# Plot the metrics for each threshold if the program finishes.  
plt.figure()
plt.plot(contaminations, accuracy, marker = 'o', color = 'red', label = 'Accuracy')
plt.plot(contaminations, precision, marker = 'o', color = 'blue', label = 'Precision')
plt.plot(contaminations, recall, marker = 'o', color = 'green', label = 'Recall')
plt.plot(contaminations, f1, marker = 'o', color = 'black', label = 'F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score Percentage (%)')
plt.legend()
plt.show()
 
