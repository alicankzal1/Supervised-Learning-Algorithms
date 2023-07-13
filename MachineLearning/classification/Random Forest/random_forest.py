import pandas as pd
import numpy as np
#%% import data

df = pd.read_csv("data.csv")
df.drop(["id", "Unnamed: 32"], axis= 1, inplace= True)
#%%
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
X = df.drop("diagnosis", axis= 1)
y = df.diagnosis.values
#%% normalization 
X = (X - np.min(X)) / (np.max(X) - np.min(X))
#%% train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state= 42)
#%% decision tree
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("decision tree score : ", dt.score(X_test, y_test))
#%% random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 100, random_state= 42)
rf.fit(X_train, y_train)
print("random forest algo result : ", rf.score(X_test, y_test))
