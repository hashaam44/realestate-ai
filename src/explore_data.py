import pandas as pd

file_path = "data/realtor-data.zip.csv"

df = pd.read_csv(file_path)

print(df.head())