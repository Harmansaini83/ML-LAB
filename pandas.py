"""As compared to Excel, Pandas is lot flexible and through Pandas we can work with big data. 
Pandas is an open-source library that is built on top of NumPy library. It is a Python package 
that offers various data structures and operations for manipulating numerical data and time series. 
It is mainly popular for importing and analyzing data much easier. Pandas is fast and it has high-performance 
& productivity for users."""


import pandas as pd   # imports the pandas library and gives it the alias 'pd'


df = pd.read_csv('pokemon.csv')   # reads a CSV file and loads it into a DataFrame named df
print(df)
print()

print(df.shape)   # returns a tuple (rows, columns) representing DataFrame dimensions
print()

df.info()         # prints summary of DataFrame: column names, non-null counts, data types, memory usage
print()

print(df.describe())   # shows statistical summary of numerical columns: mean, std, min, max, quartiles
print()

print(df.head(5))      # displays the first 5 rows of the DataFrame
print()

print(df.tail(5))      # displays the last 5 rows of the DataFrame
print()

print(df.columns)      # prints all column names of the DataFrame
print()

print(df.head())       # displays first 5 rows (default value of head() is 5)
print()

print(df.iloc[0:4])    # iloc = integer-location-based indexing; selects rows 0 to 3 (4 excluded)
print()

print(df.iloc[2,1])    # iloc[row_index, column_index]; selects element at row 2, column 1
print()

print(df.loc[df['type1'] == 'fire'])   # loc = label-based selection; filters rows where type1 column equals 'fire'
print()


pd.set_option('display.max_columns', 13)   # increases maximum number of columns displayed when printing a DataFrame
pd.set_option('display.max_rows', 800)     # increases maximum number of rows displayed for printing
print(df)
print()


cols = list(df.columns)         # converts DataFrame column names to a Python list
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]   # reorders columns: first 4, then last column, then columns 4–11
print(df)
print()
