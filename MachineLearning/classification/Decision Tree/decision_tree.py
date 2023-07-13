import pandas as pd
import numpy as np
#%%
df = pd.read_csv("data.csv")
df.drop(["id", "Unnamed: 32"], axis= 1, inplace= True)
#%%
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df.diagnosis.values
X = df.drop("diagnosis", axis= 1)
#%% normalization

X = (X - np.min(X)) / (np.max(X) - np.min(X))
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state= 42)
#%%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("score : ", dt.score(X_test, y_test))