import numpy as np 
import pandas as pd
#%% importing data

df = pd.read_csv("data.csv")
df.drop(["id", "Unnamed: 32"], axis= 1, inplace= True)
#%%
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

y = df.diagnosis.values
X = df.drop("diagnosis", axis= 1) 
#%% normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))
#%% train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state= 42)
#%% random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, random_state= 1)
rf.fit(X_train, y_train)
print("random forest algo result : ", rf.score(X_test, y_test))
#%% confusion matrix

y_pred = rf.predict(X_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
#%% confusion matrix visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize= (5,5))
sns.heatmap(cm, annot= True, linewidths= 0.5, linecolor= "red", fmt= ".0f", ax= ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()