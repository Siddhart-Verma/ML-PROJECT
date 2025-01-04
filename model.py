'''import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('student-por.csv')
print("Dataset loaded successfully!")
print(data.head())
print(data.columns)

# Features (X) and Targets (y)
X = data[['sex', 'age','Mjob', 'Fjob', 'studytime','failures', 'famsup', 'activities',
          'higher', 'internet', 'freetime']]

y = data[['G1', 'G2', 'G3', 'absences']]

# Preprocessing: Handle categorical and binary fields
binary_cols = ['famsup', 'activities','higher', 'internet']
categorical_cols = ['sex', 'Mjob', 'Fjob']

# Convert binary columns (yes/no) to 1/0
for col in binary_cols:
    X.loc[:, col] = X[col].map({'yes': 1, 'no': 0})

# Ensure target variable has consistent data types
for col in y.columns:
    if y[col].dtype == object:
        y.loc[:, col] = y[col].map({'yes': 1, 'no': 0})  # Convert yes/no to 1/0
    else:
        y.loc[:, col] = y[col].astype(float)  # Convert to float for consistency

# Preprocessing pipeline for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train and test sets.")

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

# Cross-validation
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model R^2 score: {accuracy:.2f}")

# Predict
sample_input = X_test.iloc[:5]  # Take the first 5 rows from the test set
predictions = model_pipeline.predict(sample_input)
print("Predictions on sample input:")
print(predictions)

import joblib

# After training your model pipeline, save it to a file
joblib.dump(model_pipeline, 'model_pipeline.pkl')


'''










'''import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv('student-por.csv')
print("Dataset loaded successfully!")
print("Dataset columns:", data.columns)

# Features (X) and Targets (y)
X = data[['sex', 'age', 'Mjob', 'Fjob', 'studytime', 'failures', 'activities', 'higher', 'internet', 'freetime']]
y = data[['G1', 'G2', 'G3', 'absences']]

print("Features before split:", X.columns)

# Preprocessing: Handle categorical and binary fields
binary_cols = ['activities', 'higher', 'internet']
categorical_cols = ['sex', 'Mjob', 'Fjob']

# Convert binary columns (yes/no) to 1/0
X.loc[:, binary_cols] = X.loc[:, binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Verify that all columns are correctly processed
print("Processed features:", X.head())

# Preprocessing pipeline for categorical and numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), ['age', 'studytime', 'failures', 'freetime'])
    ],
    remainder='passthrough'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train and test sets.")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Ensure column order and presence (reorder to match preprocessor expectations)
X_train = X_train.reindex(columns=categorical_cols + ['age', 'studytime', 'failures', 'freetime'] + binary_cols)
X_test = X_test.reindex(columns=categorical_cols + ['age', 'studytime', 'failures', 'freetime'] + binary_cols)

print("Reindexed X_train columns:", X_train.columns)
print("Reindexed X_test columns:", X_test.columns)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

# Cross-validation
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Check preprocessed X_train
print("Preprocessed X_train sample shape:", model_pipeline.named_steps['preprocessor'].transform(X_train).shape)

# Evaluate the model
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model R^2 score: {accuracy:.2f}")

# Predict
sample_input = X_test.iloc[:5]  # Take the first 5 rows from the test set
predictions = model_pipeline.predict(sample_input)
print("Predictions on sample input:")
print(predictions)

# Save the model pipeline
joblib.dump(model_pipeline, 'model_pipeline.pkl')
print("Model pipeline saved as 'model_pipeline.pkl'")


'''



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pymongo import MongoClient

# Load the dataset
df = pd.read_csv('student-por.csv')

# Create a new column 'success_rate'
df['success_rate'] = df[['G1', 'G2', 'G3']].mean(axis=1)

# Apply condition on absences (for example, absences < 5)
df = df[df['absences'] < 5]

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df[['G1', 'G2', 'G3', 'absences']]
y = df['success_rate']  # Use the new 'success_rate' column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['project']
collection = db['future']

# Prepare the data to be saved
predictions = pd.DataFrame({
    'G1': X_test['G1'],
    'G2': X_test['G2'],
    'G3': X_test['G3'],
    'absences': X_test['absences'],
    'predicted_success_rate': y_pred
})

# Convert DataFrame to dictionary and insert into MongoDB
collection.insert_many(predictions.to_dict('records'))

print("Predictions saved to the database successfully!")
