import pandas as pd
import numpy as np

data = {
    "Name": ["CJ", "Lance", "Angel", "Nike", "Rodito", "Jham"],
    "Age": [22, 23, 21, 23, 23, 22],
    "Salary": [30000, 32000, 31000, 30000, 30000, 500000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

print("\nMissing Values:")
print(df.isnull())

print("\nTotal Missing Values:")
print(df.isnull().sum())

df["Age"].fillna(df["Age"].mean(), inplace=True)

print("\nAfter Replacing Missing Age with Mean:")
print(df)

print("\nDuplicate Rows:")
print(df[df.duplicated()])

df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]

print("\nDetected Outliers:")
print(outliers)

df = df[(df["Salary"] >= lower_bound) & (df["Salary"] <= upper_bound)]

print("\nFinal Data after Removing Outliers:")
print(df)
