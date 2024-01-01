import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_data['Age']=train_data.Age.fillna(train_data.Age.mean())
test_data['Age'] = test_data.Age.fillna(test_data.Age.mean())

#Train data

y = train_data["Survived"]

features = ["Pclass", "Sex", "Parch", "Age", "SibSp"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("predictions.csv", index=False)

#Show data

print(train_data.head())

men = train_data.loc[train_data.Sex =='male']['Survived']
rate_men = sum(men) /len(men)

women = train_data.loc[train_data.Sex =='female']['Survived']
rate_women = sum(women) /len(women)

under18 = train_data.loc[train_data.Age < 18]['Survived']
rate_under18 = sum(under18) /len(under18)

lowerClass = train_data.loc[train_data.Pclass == 3]['Survived']
rate_lowerClass = sum(lowerClass) /len(lowerClass)

upperClass = train_data.loc[train_data.Pclass == 1]['Survived']
rate_upperClass = sum(upperClass) /len(upperClass)

xValues = np.array(["men", "women", "under 18s", "lower class", "upper class"])
yValues = np.array([rate_men, rate_women, rate_under18, rate_lowerClass, rate_upperClass])
colours = np.array(["steelblue", "indianred", "lightpink", "mediumpurple", "lightgreen"])

plt.figure(figsize=(6, 5))

data = {"Demographic": xValues,
        "Survival Rate": yValues}
df = pd.DataFrame(data, columns=["Demographic", "Survival Rate"])

plots = sns.barplot(x="Demographic", y="Survival Rate", data=df, palette=sns.color_palette(colours))
 
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), ".2f")+"%", 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha="center", va="center",
                   size=9, xytext=(0, 8),
                   textcoords="offset points")
    
plt.title("Titanic Survival Rates", size=12)
plt.show()
