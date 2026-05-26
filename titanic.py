import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")




print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable (Survived):")
print(df["Survived"].value_counts())






df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
)
df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
df["Title"] = df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})
df["Title"] = df["Title"].fillna(0)


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))


df["Age*Class"] = df["Age"] * df["Pclass"]


df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())


df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)


df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})



X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "Title", "FamilySize", "IsAlone", "Age*Class", "FareBin", "AgeBin"]]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")


new_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 1,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0,
    "Title": 0,
    "FamilySize": 1,
    "IsAlone": 1,
    "Age*Class": 25 * 3,
    "FareBin": 0,
    "AgeBin": 2
}])

new_passenger_scaled = scaler.transform(new_passenger)
prediction = model.predict(new_passenger_scaled)[0]
probability = model.predict_proba(new_passenger_scaled)[0]

print("\nNew Passenger Prediction:")
print("Result:", "Survived" if prediction == 1 else "Did not survive")
print("Survival probability:", round(probability[1] * 100, 2), "%")