# student_predictor

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
!pip install pydotplus
#Import libraries

import pandas as pd
import numpy as np

Needed for Decision Tree Visualization
import pydotplus
from IPython.display import Image

Needed for Logistic Regression and Decision Tree Models
from pathlib import Path
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris (used for decision tree visualization)
%matplotlib inline

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

Machine Learning Library and Visualizations
from yellowbrick.features import Rank1D
from yellowbrick.features import Rank2D 
from yellowbrick.features import ParallelCoordinates
from yellowbrick.classifier import ClassificationReport
from yellowbrick.features import RadViz
from yellowbrick.features import JointPlotVisualizer
from yellowbrick.features import PCADecomposition
from yellowbrick.features import FeatureImportances


Data Visualization: creating graphs and other visualizations to show the logistic and decision tree models/features.

Dataset Variables
Can someone include screenshots from the UCI website about the variables? website:https://archive.ics.uci.edu/dataset/320/student+performance



