# Importing necessary libraries
import pandas as pd          # pandas is used for handling and analyzing structured data (CSV, Excel, etc.)
import numpy as np           # numpy is used for numerical operations and handling arrays

# Read the dataset from a CSV file
df = pd.read_csv('Social_Network_Ads.csv')   # pd.read_csv() loads CSV file into a DataFrame

# Display the first 5 rows of the dataset
print(df.head())          # df.head() shows the top 5 rows to preview the data

# Select the input features (independent variables)
X = df[['Age', 'EstimatedSalary']]   # Selecting only two features from the dataset

# Select the output variable (dependent variable)
y = df["Purchased"]                  # The column we want to predict

# Import train-test split function
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
# test_size=0.3 means 30% data goes to testing, 70% to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Check the shape (rows, columns) of the training data
X_train.shape     # Example: (280, 2)

# Check for missing values in each column of X_train
X_train.isna().sum()    # isna() returns True for missing values; sum() counts them

# Import the SimpleImputer for handling missing values
from sklearn.impute import SimpleImputer

# Create an imputer object
# missing_values=np.nan → look for NaN values
# strategy='mean' → replace missing values with column mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Learn the mean of each column in X_train (fit = calculate statistics)
imputer.fit(X_train)

# Replace missing values in X_train using the learned means
X_train_imputed = imputer.transform(X_train)

# Apply the same transformation to test data (use same mean, no refitting)
X_test_imputed = imputer.transform(X_test)



#Use feature Scaling to convert different scales to a standard scale to make it easier for Machine learning algorithms
# Import preprocessing tools from sklearn
from sklearn import preprocessing     # General preprocessing utilities
from sklearn.preprocessing import MinMaxScaler   # MinMaxScaler is used for normalization/scaling

# Create a MinMaxScaler object
# MinMaxScaler converts values to a range between 0 and 1
# Formula: (x - min) / (max - min)
# Useful when features have very different scales: 
# Example: Age (18–60), Salary (15,000–150,000)
scalar = MinMaxScaler()

# Learn the minimum and maximum values of each column from the training data
# fit() calculates the min and max for each feature
scalar.fit(X_train_imputed)

# Transform the training data using the learned min/max
# transform() applies the scaling to convert all values into the new range
X_train_scaled = scalar.transform(X_train_imputed)

# Apply the same scaling to the test data (using the SAME scaling learned from training)
# Never fit() on test data → only transform()
X_test_scaled = scalar.transform(X_test_imputed)

# Display the scaled training data
# Now all values lie between 0 and 1
X_train_scaled



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(model.score(X_test_scaled,y_test))
