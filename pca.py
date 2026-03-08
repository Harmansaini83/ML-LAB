# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB # Import Gaussian Naive Bayes
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv("Final_data.csv")

# Step 2: Separate features (X) and target (y)
X = data.drop(columns=['Burns_Calories_Bin'])
y = data['Burns_Calories_Bin']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Step 3: Identify categorical and numerical columns
categorical_features = ['Gender']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Step 4: Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Step 5: Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Step 6: Create a pipeline that first preprocesses the data and then trains the Naive Bayes model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GaussianNB())]) # Use GaussianNB

# Step 7: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 8: Train the model
pipeline.fit(X_train, y_train)

# Step 9: Predict and evaluate accuracy
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 10: Perform PCA (reduce to 2 dimensions for visualization) - Apply preprocessing first
# Need to apply the same preprocessing to the data before PCA
X_processed = preprocessor.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plot PCA result
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='rainbow')
plt.title("PCA - 2D Visualization of Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
