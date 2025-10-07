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
plt.figure(figsize=(8,6))
plt.hist(train_df['Step'],bins=13,edgecolor='black')
plt.xlabel("Step number")
plt.ylabel("Frequency")
plt.title("Distribution of Data points per Step (Training Data)")
plt.show()

# Scatter plots of all coordinates with Step Numbers
plt.figure(figsize=(8,6))
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

# Pairwise Relationship of Coordinates by Step
pair_plot= sbn.pairplot(train_df,
                        vars=['X','Y','Z'],
                        hue="Step",
                        palette=sbn.color_palette("husl",13),
                        diag_kind='hist')
pair_plot.fig.suptitle("Pairwise Relationships of (X,Y,Z) Coordinates by Step",y=1.03)
plt.show()

# 3-D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111,projection='3d')

# Scatter plot with color-coding by Step
sc = ax.scatter(
    train_df['X'], train_df['Y'], train_df['Z'],
    c=train_df['Step'],
    s=30, alpha=0.8
)
# Label axes and Title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of (X, Y, Z) Coordinates by Step')

# Add colorbar as legend
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Step Number')
plt.show()

# Step 3: Correlation Analysis

corr_matrix = train_df.corr(method='pearson')
print("\nCorrelation Matrix:\n\n", corr_matrix,"\n")
plt.figure(figsize=(8,6))
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

# Model 1 - Support Vector Machine Classifier (GridSearch CV)

# SVC Model Pipeline
pipe1 = Pipeline([
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
gs_clf1 = GridSearchCV(pipe1,param_grid_clf1,scoring='f1_macro',cv=5,n_jobs=-1)

# Running grid search on training data
gs_clf1.fit(X_train,y_train)
print("Best Parameters for SVC:", gs_clf1.best_params_,"\n")
print("Best Cross-Validation F1 Score (SVM):", gs_clf1.best_score_)

clf1 = gs_clf1.best_estimator_
print("SVM Training Accuracy:", clf1.score(X_train,y_train))
print("SVM Test Accuracy:", clf1.score(X_test,y_test),"\n")

# Model 2 - Logistic Regression Classifier (GridSearch CV)

# LR Model Pipeline
pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500,random_state=42))
    ])
param_grid_clf2 =  {
        'clf__C': [0.1,0.5,1,5,10],
        'clf__solver': ['lbfgs','newton-cg'],
        'clf__class_weight': [None,'balanced']
        }
# GridSearch Call
gs_clf2 = GridSearchCV(pipe2,param_grid_clf2,scoring='f1_macro',cv=5,n_jobs=-1)

# Running GridSearch on training data
gs_clf2.fit(X_train,y_train)
print("Best Parameters for LR model:", gs_clf2.best_params_,"\n")
print("Best Cross-Validation F1 Score (LR):", gs_clf2.best_score_)

clf2 = gs_clf2.best_estimator_
print("LR Training Accuracy:", clf2.score(X_train,y_train))
print("LR Test Accuracy:", clf2.score(X_test,y_test),"\n")

# Model 3 - Decision Tree Classifier (GridSearch CV)

# Pipeline
pipe3 = DecisionTreeClassifier(random_state=42)

# Parameter Grid
param_grid_clf3 = {
        'max_depth': [None,3,7,10],
        'min_samples_split': [2,5,10,20],
        'min_samples_leaf' : [1,2,4,8],
        'criterion': ['gini', 'entropy','log_loss'],
        'max_features': [None,'sqrt', 'log2'],
        'class_weight': [None,'balanced']
        }
# GridSearch Call
gs_clf3 = GridSearchCV(pipe3,param_grid_clf3,scoring='f1_macro',cv=5,n_jobs=-1)

# Running GridSearch on training data
gs_clf3.fit(X_train,y_train)
print("Best Parameters for DT model:", gs_clf3.best_params_,"\n")
print("Best Cross-Validation F1 Score (DT):", gs_clf3.best_score_)

# Save the best model with the optimal hyperparameters
clf3 = gs_clf3.best_estimator_

# Best Model accuracy score
print("DT Training Accuracy:", clf3.score(X_train,y_train))
print("DT Test Accuracy:", clf3.score(X_test,y_test),"\n")

# Model 4 - Random Forest Classifier (RandomizedSearch CV)
from scipy.stats import randint

# Pipeline
pipe4 = RandomForestClassifier(random_state=42)

# Parameter Grid
param_rand_clf4 = {
    'n_estimators': randint(100,400),
    'max_depth': randint(5,15),
    'min_samples_split': randint(2,20),
    'min_samples_leaf': randint(1,8),
    'max_features': [None,'sqrt', 'log2'],
    'criterion': ['gini','entropy'],
    'class_weight': [None,'balanced']
    }

# RandomizedSearch Call
rs_clf4 = RandomizedSearchCV(pipe4,param_rand_clf4,scoring='f1_macro',
                             n_iter=40,cv=5,n_jobs=-1,random_state=42)

# Running GridSearch on training data
rs_clf4.fit(X_train,y_train)

# Display Results
print("Best Parameters for RF model:", rs_clf4.best_params_,"\n")
print("Best Cross-Validation F1 Score (RF):", rs_clf4.best_score_)

# Save the best model with the optimal hyperparameters
clf4 = rs_clf4.best_estimator_

# Best Model accuracy score
print("RF Training Accuracy:", clf4.score(X_train,y_train))
print("RF Test Accuracy:", clf4.score(X_test,y_test),"\n")

# Summarize the models
models = {
    "SVC": clf1,
    "Logistic Regression": clf2,
    "Decision Tree": clf3,
    "Random Forest": clf4
    }
# Step 5: Model Performance Analysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

results = []
labels = sorted(y_test.unique())
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test,y_pred,average='weighted')
    rec = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 score': f1
        })
    
    plt.figure(figsize=(8,6))
    sbn.heatmap(cm,annot=True,cmap="Reds",xticklabels=labels,yticklabels=labels,cbar=True)
    plt.title(f"{name}- Confusion Matrix")
    plt.xlabel("Predicted Step")
    plt.ylabel("Actual Step")
    plt.show()

results_df=pd.DataFrame(results)
results_df=results_df.sort_values(by='F1 score', ascending=False).reset_index(drop=True)
print("\nModel Perfomance Summary:\n", results_df)

# Step 6: Stacked Model Performance
from sklearn.ensemble import StackingClassifier
meta_model = LogisticRegression(max_iter=2500, random_state=42)

stacked_model = StackingClassifier([('svc',clf1),('rf',clf4)],meta_model,cv=5,n_jobs=-1)

stacked_model.fit(X_train, y_train)

y_pred_stack = stacked_model.predict(X_test)

acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test,y_pred_stack,average='weighted')
rec_stack = recall_score(y_test, y_pred_stack,average='weighted')
f1_stack = f1_score(y_test, y_pred_stack, average='weighted')
cm_stack = confusion_matrix(y_test, y_pred_stack, labels=labels)

print("\nStacked Model Performance:")
print(f"\nAccuracy: {acc_stack:.4f}")
print(f"\nPrecision: {prec_stack:.4f}")
print(f"\nRecall: {rec_stack:.4f}")
print(f"\nF1(macro): {f1_stack:.4f}")

plt.figure(figsize=(8,6))
sbn.heatmap(cm_stack,annot=True,cmap="Blues",xticklabels=labels,yticklabels=labels,cbar=True)
plt.title("Stacked Confusion Matrix")
plt.xlabel("Predicted Step")
plt.ylabel("Actual Step")
plt.show()

# Step 7: Model Evaluation
import joblib

joblib.dump(clf2,"Best_Step_Classifier.joblib")
print("Model Saved successfully")

loaded_model = joblib.load("Best_Step_Classifier.joblib")
print("Model Loaded Successfully")

eval_data = pd.DataFrame([
    [9.375,3.0625,1.51],
    [6.995,5.125,0.3875],
    [0,3.0625,1.93],
    [9.4,3,1.8],
    [9.4,3,1.3]], columns=['X','Y','Z'])

predicted_step = loaded_model.predict(eval_data)

for i, row in eval_data.iterrows():
    print(f"\nPredicted Maintenance Step for coordinates {row.values.tolist()} is: Step {predicted_step[i]}")