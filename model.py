from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import   StandardScaler as SS
from sklearn.ensemble import RandomForestClassifier as RFC
import pickle
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# train test split
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=0)

# scaling the data
scaler = SS()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# modelling
classifier = RFC()
classifier.fit(X_train, y_train)

# making a pickle file 
pickle.dump(classifier, open("model.pkl", "wb"))

