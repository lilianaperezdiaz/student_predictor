# Student Success Predictor

## Overview
A machine learning project that predicts academic achievement in Portuguese secondary education (high school) by analyzing external factors that influence student performance. The project focuses on two core subjects:
- Mathematics
- Portuguese Language

The analysis is based on comprehensive datasets from two Portuguese schools, incorporating student grades, demographic information, and social factors collected through school reports and questionnaires.

## Data Source
The dataset is sourced from the UCI Machine Learning Repository (December 2024) and was originally analyzed by Cortez and Silva (2008). 

### Important Note About Grades
The target variable G3 (final grade) shows strong correlation with G1 (first period) and G2 (second period) grades. While including G1 and G2 improves prediction accuracy, predicting G3 without these intermediate grades provides more practical value for early intervention.

## Features
The dataset includes multiple categories of predictive features:
- Student Demographics
- Family Background
- Academic History
- Social Factors
- School-Related Variables

*Note: Detailed variable descriptions and screenshots from UCI will be added upon availability*

## Methodology

### Machine Learning Approach
The project implements two primary models:
1. Logistic Regression
   - Establishes linear relationships between features and academic outcomes
   - Suitable for binary classification of student success
   
2. Decision Tree Classification
   - Hierarchical decision-making structure
   - Components:
     - Root Node: Initial dataset split
     - Internal Nodes: Feature-based decisions
     - Branches: Decision pathways
     - Leaf Nodes: Final predictions

### Feature Analysis
- Comprehensive feature ranking and importance assessment
- Correlation analysis between predictors
- Feature selection based on statistical significance

## Technical Implementation

### Required Libraries

#### Core Data Processing
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

#### Model Implementation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
```

#### Model Evaluation
```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
```

#### Data Enhancement
```python
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

#### Visualization Tools
```python
from yellowbrick.features import (
    Rank1D,
    Rank2D,
    ParallelCoordinates,
    RadViz,
    JointPlotVisualizer,
    PCADecomposition,
    FeatureImportances
)
from yellowbrick.classifier import ClassificationReport
```

#### Decision Tree Visualization
```python
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz, plot_tree
```

## Project Structure
```
student-success-predictor/
├── data/
│   ├── student-mat.csv      # Mathematics performance dataset
│   └── student-por.csv      # Portuguese language dataset
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── exploratory_analysis.ipynb
│   └── model_development.ipynb
├── src/
│   ├── features/
│   ├── models/
│   └── visualization/
└── README.md
```

## Getting Started
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run Jupyter notebooks in the `notebooks/` directory

## Future Improvements
- Implementation of additional machine learning models
- Feature engineering for improved prediction accuracy
- Development of a web interface for real-time predictions

## References
- Original paper: Cortez and Silva (2008)
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)


