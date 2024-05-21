import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
data= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv")
print(data.head())
print(data.describe())
print(data.isnull().sum())
plt.figure(figsize=(15, 10))
plt.title("Product Purchased By People Through Social Media Marketing")
sns.histplot(x="Age", hue="Purchased", data= data)
plt.show()
plt.title("Product purchased By People Based on their Salary")
sns.histplot(x="EstimatedSalary", hue="Purchased", data= data)
plt.show()
x=np.array(data[["Age", "EstimatedSalary"]])
y=np.array(data[["Purchased"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.10, random_state=42 )
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
Predictions = model.predict(xtest)
print(classification_report(ytest, Predictions))
