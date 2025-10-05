
# Step 1: Data Processing
import pandas as pd
data = pd.read_csv("Project 1 Data.csv")
print(data.columns)

#Defining x and y variables to distinguish between target and feature variables
x = data[['X','Y','Z']]
y = data['Step']

# Data Splitting before proceeding to next step in order to aviod any data snooping
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as sss
# Define the parameters of the splitter and perform a 20-80 split of the data
splitter= sss(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in splitter.split(x,y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]