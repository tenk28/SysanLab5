import pandas as pd

df = pd.ExcelFile("km.xlsx")
df = pd.read_excel(df, sheet_name="O1I1")
print(df)
print()
print(df.dtypes)
