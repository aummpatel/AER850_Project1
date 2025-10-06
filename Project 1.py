# Importing required general libraries

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

# Import all the required modules and classes for model development
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Variable Selection -
# Features
X_train = train_df[['X','Y','Z']] 
X_test = test_df[['X','Y','Z']]
# Target
y_train = train_df['Step']
y_test = test_df['Step']

# Model 1 - Support Vector Machine Classifier (GridSearch Optimized)
# SVC Model Pipeline
pipesvc = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
        ])
    
# Parameter Grid to search
param_grid_clf1 = {
        'clf__C': [0.1,0.5,1,5,10],
        'clf__kernel': ['linear','rbf','poly'],
        'clf__gamma': ['scale','auto'],
        'clf__class_weight': [None,'balanced']
        }
    
# GridSearch Cross-Validation using f1_macro as the scoring metric to determine the best hyperparameters
gs_clf1 = GridSearchCV(pipesvc,param_grid_clf1,scoring='f1_macro',cv=5,n_jobs=-1)
# Running grid search on training data
gs_clf1.fit(X_train,y_train)

print("Best Parameters for SVC:", gs_clf1.best_params_)
print("Best Cross-Validation F1 Score:", gs_clf1.best_score_)

clf1 = gs_clf1.best_estimator_
print("SVM Training Accuracy:", clf1.score(X_train,y_train))
print("SVM Training Accuracy:", clf1.score(X_test,y_test))

    
    # Model 2 - Random Forest Classifier (GridSearch Optimized)
    
    # Model 3 - Decision Tree Classifier (GridSearch Optimized)
    
    # Model 4 - Logistic Regression Classifier (RandomizedSearch Optimized)
    

# Step 5: Model Performance Analysis

# Step 6: Stacked Model Performance

# Step 7: Model Evaluation





