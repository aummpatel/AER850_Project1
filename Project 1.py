# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn  

# Step 1: Data Processing
# Read the .csv into a DataForm variable
df= pd.read_csv("Project 1 Data.csv")

# Data Splitting before proceeding to next step in order to aviod any data snooping
from sklearn.model_selection import StratifiedShuffleSplit as sss

# Define the parameters of the splitter to perform a 20-80 split of the data
splitter= sss(n_splits=1,test_size=0.2,random_state=42)

# Stratification on 'Step' ensures that the proportion of each step is preserved in both datasets
for train_index, test_index in splitter.split(df,df['Step']):
    train_df = df.loc[train_index].reset_index(drop=True)
    test_df = df.loc[test_index].reset_index(drop=True)

# Step 2: Data Visualization

# Histogram of 'Step' showing class distribution of the training dataset
plt.hist(train_df['Step'],bins=13,edgecolor='black')
plt.xlabel("Step number")
plt.ylabel("Frequency")
plt.title("Distribution of Data points per Step (Training Data)")
plt.show()

# Scatter plots of all coordinates with Step Numbers
fig, axes = plt.subplots(3, 1,layout='constrained')
axes[0].scatter(train_df['Step'], train_df['X'], edgecolors="black", color="red")
axes[0].set_ylabel("X Coordinate")
axes[0].set_title("Scatter Plot of Coordinates vs Step Number")
axes[1].scatter(train_df['Step'], train_df['Y'], edgecolors="black", color="cyan")
axes[1].set_ylabel("Y Coordinate")
axes[2].scatter(train_df['Step'], train_df['Z'], edgecolors="black", color="yellow")
axes[2].set_xlabel("Step Number")
axes[2].set_ylabel("Z Coordinate")
plt.show()

# Step 3: Correlation Analysis

corr_matrix = train_df.corr(method='pearson')
print("Correlation Matrix:\n", corr_matrix,"\n")
sbn.heatmap(np.abs(corr_matrix))
plt.title('Pearson Correlation Heatmap')
plt.show()

# Step 4: Classification Model Development

# Step 5: Model Performance Analysis

# Step 6: Stacked Model Performance

# Step 7: Model Evaluation





