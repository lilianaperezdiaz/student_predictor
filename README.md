# Student Predictor 

Project Purpose

The purpose of this project is to predict academic achievement in secondary education(high school) of two Portuguese schools. This project explores the external factors that may support or hinder academic achievement.

Student Peformance

This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details).(UCI, December 2024.)

Methodology

Machine Learning:using machine learning models to analyze historical data. This allows data professionals to predict or make decisions without knowing the final outcome. Academic Achievement (project) will use the logistic regression and decision tree models.

Model Analysis:Logistic regression-using features, it applies a linear relationship between features (independent variables) and targets.

Decision Tree: Root Nodes: entire dataset and splits based on feature significance. Internal Nodes: decision based on features and decision splits the dataset into smaller subsets. Branches: connect nodes and showcase the outcomes. Leaf nodes: terminal node to the final prediction.

Feature Analysis: features are independent variables used in the model.

Libraries:

Model Evaluation, Feature Ranking and Analysis:
1)!pip install pydotplus

2)import pandas as pd
3)import numpy as np

Needed for Decision Tree Visualization
4)import pydotplus
5)from IPython.display import Image

Needed for Logistic Regression and Decision Tree Models
6)from pathlib import Path
7)from sklearn import tree
8)from sklearn.tree import export_graphviz
9)from sklearn.datasets import make_classification
10)from sklearn.model_selection import train_test_split, cross_val_score
11)from sklearn.ensemble import RandomForestClassifier
12)from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
13)from imblearn.over_sampling import SMOTE
14)from sklearn.utils.class_weight import compute_class_weight
15)from sklearn.preprocessing import MinMaxScaler,StandardScaler
16)from sklearn.linear_model import LogisticRegression 
17)from sklearn.tree import DecisionTreeClassifier
18)from sklearn.datasets import load_iris (used for decision tree visualization)
%matplotlib inline

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

Machine Learning Library and Visualizations
19)from yellowbrick.features import Rank1D
20)from yellowbrick.features import Rank2D 
21)from yellowbrick.features import ParallelCoordinates
22)from yellowbrick.classifier import ClassificationReport
23)from yellowbrick.features import RadViz
24)from yellowbrick.features import JointPlotVisualizer
25)from yellowbrick.features import PCADecomposition
26)from yellowbrick.features import FeatureImportances


Data Visualization: creating graphs and other visualizations to show the logistic and decision tree models/features.

Dataset Variables
Can someone include screenshots from the UCI website about the variables? website:https://archive.ics.uci.edu/dataset/320/student+performance



