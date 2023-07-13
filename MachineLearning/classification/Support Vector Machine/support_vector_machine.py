# import librarires

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%% data

df = pd.read_csv("data.csv")
df.head()
#%% drop
df.drop(["id", "Unnamed: 32"], axis= 1, inplace= True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df.diagnosis.values
X = df.drop(["diagnosis"],axis=1)
#%%normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))
#%% svm model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
#%% 
svm = SVC(random_state= 1)
svm.fit(X_train, y_train)
print("accuracy of svm algo : ", svm.score(X_test, y_test))