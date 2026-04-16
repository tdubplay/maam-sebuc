import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.DataFrame({
     "Name": ["Alice", "Bob"],
     "Age": [25,30],
     "City": ["NYC", "LA"]
})

print("df.shape:", df.shape)
print("\ndf.info():")
df.info()
print("\ndf.head():")
print(df.head())
print("\ndf.describe():")
print(df.describe())
 
df = pd.DataFrame({
"Name": ["Alice", "Bob"],
"Age": [25, 30],
"City": ["NYC", "LA"]
})

print("df['Name']: \n", df["Name"])
print("\ndf[['Name', 'Age']]:\n", df [["Name", "Age"]])

print("\ndf[df['Age'] > 25]:\n", df [df ["Age"] > 25])
print("\ndf[df['City'] == 'NYC']:\n", df[df["City"] == "NYC"])

df = pd.DataFrame({
"Name": ["Alice", "Bob"],
"Age": [25, 30],
"City": ["NYC", "LA"]
})

df.loc[df["Name"]== "Bob", "Age"] = 32

df["Bunos"] = df ["Age"] * 0.10

print(df)

s = pd.Series([0.1, 0.2, 0.3, 0.4], index=['a', 'b', 'c', 'd'])
print("Pandas Series:\n", s)

data = {
    'FeatureA': [1.0, 2.5, 3.1, 4.7],
    'FeatureB': ['cat', 'dog', 'cat', 'rabbit'],
    'Target': [0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("\nPandas DataFrame:\n", df)

try:

    with open('data.csv', 'w') as f:
        f.write("ID,Temperature,Humidity\n")
        f.write("1,25.5,60\n")
        f.write("2,26.1,62\n")
        f.write("3,24.9,58\n")
        f.write("4,,55\n")

    df_from_csv = pd.read_csv('data.csv')

    print("\nDataFrame loaded from CSV:\n", df_from_csv)

except FileNotFoundError:
    print("\nError: data.csv not found.")

except Exception as e:
    print(f"\nAn error occurred reading CSV: {e}")

temperature = df_from_csv['Temperature']
print("\nTemperature Series:\n", temperature)

subset_df = df_from_csv[['ID', 'Humidity']]
print("\nSubset DataFrame:\n", subset_df)

print("\nRow with index 1 (.loc):\n", df_from_csv.loc[1])
print("\nFirst row (.iloc):\n", df_from_csv.iloc[0])

high_humidity_df = df_from_csv[df_from_csv['Humidity'] > 60]
print("\nRows with Humidity > 60:\n", high_humidity_df)

print("\nMissing values per column:\n", df_from_csv.isnull().sum())

df_dropped = df_from_csv.dropna()
print("\nDataFrame after dropping NaN rows:\n", df_dropped)

mean_temp = df_from_csv['Temperature'].mean()
df_filled = df_from_csv.fillna({'Temperature': mean_temp})
print("\nDataFrame after filling NaN temperature with mean:\n", df_filled)

data = np.random.randn(100, 3)

df = pd.DataFrame(data, columns=['A', 'B', 'C'])

result = df[df['A'] > 0].groupby('B').mean()

print(result)

data = {'ID': [1, 2, 3, 4, 5], 'Temperature': [22.5, 23.0, 21.8, 24.2, 23.5]}
df_filled = pd.DataFrame(data)

df_plot = df_filled

plt.figure(figsize=(8, 4))

plt.plot(df_plot['ID'], df_plot['Temperature'], color='#2ca02c', marker='o', linestyle='-', linewidth=2)

plt.title('Temperature Trend Over ID', fontsize=14)
plt.xlabel('ID', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7) 

plt.tight_layout()
plt.show()

data = {
    'ID': range(1, 11),
    'Temperature': [22, 23, 21, 24, 25, 22, 23, 26, 24, 25],
    'Humidity': [45, 48, 50, 55, 60, 58, 52, 49, 47, 46]
}

df_plot = pd.DataFrame(data)

plt.figure(figsize=(8, 4))
sns.histplot(data=df_plot, x='Humidity', kde=True, color='skyblue')
plt.title('Humidity Distribution')
plt.show()

data = {
    'Temperature': [20, 21, 19, 22, 25, 28, 30, 27, 24, 22],
    'Humidity':    [55, 54, 58, 50, 45, 40, 35, 38, 42, 50]
}

df_plot = pd.DataFrame(data)

os.makedirs('static/images/numpy-pandas', exist_ok=True)

plt.figure(figsize=(8, 4))
sns.scatterplot(data=df_plot, x='Temperature', y='Humidity', s=100, color='coral')

plt.title('Temperature vs. Humidity')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('static/images/numpy-pandas/temp_vs_humidity_scatter.png')
plt.show()

df_plot['Category'] = ['A','A','A','A','A','B','B','B','B','B']

plt.figure(figsize=(8,4))
sns.boxplot(data=df_plot, x='Category', y='Temperature')
plt.title('Temperature Distribution by Category')
plt.show() 
 
