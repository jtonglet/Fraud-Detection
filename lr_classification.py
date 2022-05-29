#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


import warnings
warnings.filterwarnings('ignore')

#Set constant value
SEED = 42

#Load data
data = pd.read_csv('processed_data.csv')

y = data['fraud'].apply(lambda row : 1 if row == 'Y' else 0)
X = data.drop(columns = 'fraud')
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    stratify = y,
                                                    random_state = SEED)

#Preprocessing functions
#Categorical features are one hot encoded 
#Numerical features are scaled with Min Max Scaler
categorical_features = X_train.select_dtypes(include=['object']).columns.to_list()
numeric_features = X_train.select_dtypes(include=['int64','float64']).columns.to_list()
bool_features = X_train.select_dtypes(include=['bool']).columns.to_list()
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown= 'ignore')  
preprocessor = ColumnTransformer( 
    transformers=[
        ("num", numeric_transformer, numeric_features),  
        ("cat", categorical_transformer, categorical_features)     
    ],
    remainder = 'passthrough'   
)

#Logistic Regression Pipeline
lr = LogisticRegression(class_weight = 'balanced',
                        solver = 'saga',
                        penalty = 'elasticnet', 
                        n_jobs = -1, 
                        l1_ratio = 0.85,
                        max_iter = 200,
                        fit_intercept = False,
                        random_state = SEED)


pipe_lr = Pipeline([('preprocessing',preprocessor),
                    ('classifier',lr)])
pipe_lr.fit(X_train,y_train)


pred_train = pipe_lr.predict(X_train)
pred_test = pipe_lr.predict(X_test)
print('----Results----')
print('Confusion Matrix Test Set')
print(confusion_matrix(y_test,pred_test))
print(' ')
print('Classification Report Test Set')
print(classification_report(y_test,pred_test))
print('AUC Score Test Set : %s'%roc_auc_score(y_test,pred_test))





