William King
Fraud Detection Using Machine Learning Algorithms (3rd Year Undergrad Project)
Last Updated: July 27th 2020

About:
- Both unsupervised and supervised machine learning algorithms are used on two different contexts of fraud. 
- The performances of each model is explored with the use of performance metrics, permutation importance and 
  PDPBox plots.
- A visualisation tool is then built to help visualise and interact with each model's predictions.

This directory holds:
- DataSets:	   Folder holding the data sets used for the project.
- Metric Drops:	   Folder holding the metric drop implementation for each unsupervised algorithm (not essential).
- Audit Data: 	   Jupyter notebook holding the PROJECT IMPLEMENTATION for Audit data set.
- kmeans_predict:  A seperate file used to assist the prediction stage for kMeans.
- Warranty Claims: Jupyter notebook holding the PROJECT IMPLEMENTATION for Warranty Claims data set.

Additional Notes about the Visualisation Tool:
 - For the Jupyter notebooks, you can interact with the visualisation tools without 'running' 
   the code (scroll to the end of the Notebook).
 - The visualisation tool will show the most recent predictions of each model.
 - There is no method of 'resetting' the data points on the visualisation tool once the data has 
   been filtered by the drop down menus. It's recommended to create a copy of a Notebook and 
   use the copy instead.
 - You can zoom in and out by placing your mouse on a graph and scroll.
 - You can also drag the graphs.

UPDATES SINCE PROJECT SUBMISSION:
 - Added dynamic axis selection for visualisation tool.
 - Added PCA axis option for visualisation tool.
