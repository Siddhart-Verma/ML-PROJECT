'''from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymongo
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize the Flask app
app = Flask(__name__)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["student_db"]  # Replace with your DB name
collection = db["student_data"]  # Replace with your collection name


# Function to clean the dataset by removing specified fields
def clean_dataset():
    # Fields to remove from the dataset
    fields_to_remove = ['address', 'Pstatus', 'reason', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc','guardian', 'famsize', 'Medu',
                        'Fedu', 'guardian', 'traveltime', 'paid', 'goout', 'health']

    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)
    print(f"Fields removed from {result.modified_count} documents.")

    # Removing duplicate documents (if any exist despite _id being unique)
    pipeline = [
        {"$group": {
            "_id": "$_id",  # Group by the unique _id field
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}}  # Find groups where count > 1
    ]

    duplicates = collection.aggregate(pipeline)
    for doc in duplicates:
        ids_to_delete = doc['ids'][1:]  # Keep one document, remove the rest
        collection.delete_many({"_id": {"$in": ids_to_delete}})
    print("Duplicate documents removed, if any.")

# Clean the dataset
clean_dataset()


# Load the trained machine learning model
model_pipeline = joblib.load('model_pipeline.pkl')


# Route for the index page
@app.route('/')
def index():
    with open('index.html') as file:
        return file.read()

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form data
        name = request.form['name']
        sex = request.form['sex']
        age = int(request.form['age'])
        study_time = int(request.form['studytime'])
        fail_number = int(request.form['fail-number'])
        activities = request.form['activities']
        higher = request.form['higher']
        internet = request.form['internet']
        free_time = int(request.form['freetime'])
        Mjob = request.form.get('Mjob', 'other')
        Fjob = request.form.get('Fjob', 'other')

        # Prepare the data for prediction
        student_data = {
            "name": name,
            "sex": sex,
            "age": age,
            "study_time": study_time,
            "fail_number": fail_number,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "free_time": free_time
        }
        
        
        # Map job categories to integers (or some other suitable mapping)
        Mjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }
        Fjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }
        

        # Insert data into MongoDB collection
        collection.insert_one(student_data)

        # Prepare the data for the model (make sure the data is formatted correctly)
        input_data = {
            "sex_male": 1 if sex == 'male' else 0,
            "age": age,
            "study_time": study_time,
            "fail_number": fail_number,
            "activities": 1 if activities else 0,
            "higher_yes": 1 if higher == 'yes' else 0,
            "internet_yes": 1 if internet == 'yes' else 0,
            "free_time": free_time,
            "Mjob": Mjob_map.get(Mjob, 0),
            "Fjob": Fjob_map.get(Fjob, 0)
        }
        
        # Insert data into MongoDB collection
        collection.insert_one(student_data)

        # Convert input_data to DataFrame and apply feature scaling (if necessary)
        input_df = pd.DataFrame([input_data])
        
        # You can scale your features here if the model requires it
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)  # Scaling the input data

        # Make prediction
        prediction = model_pipeline.predict(input_scaled)

        # Return the prediction result
        result = "Pass" if prediction[0] == 1 else "Fail"
        return jsonify({"result": result, "student_data": student_data})

if __name__ == '__main__':
    app.run(debug=True)
    '''
    
    





'''from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymongo
import joblib

# Initialize the Flask app
app = Flask(__name__)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["project"]  # Replace with your DB name
collection = db["future"]  # Replace with your collection name



# Function to clean the dataset by removing specified fields
def clean_dataset():
    # Fields to remove from the dataset
    fields_to_remove = ['address', 'Pstatus', 'reason', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc']

    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)
    print(f"Fields removed from {result.modified_count} documents.")

    # Removing duplicate documents (if any exist despite _id being unique)
    pipeline = [
        {"$group": {
            "_id": "$_id",  # Group by the unique _id field
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}}  # Find groups where count > 1
    ]

    duplicates = collection.aggregate(pipeline)
    for doc in duplicates:
        ids_to_delete = doc['ids'][1:]  # Keep one document, remove the rest
        collection.delete_many({"_id": {"$in": ids_to_delete}})
    print("Duplicate documents removed, if any.")

# Clean the dataset
clean_dataset()



# Load the trained machine learning model
model_pipeline = joblib.load('model_pipeline.pkl')

# Route for the index page
@app.route('/')
def index():
    with open('index.html') as file:
        return file.read()

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form data
        name = request.form['name']
        sex = request.form['sex']
        age = int(request.form['age'])
        study_time = int(request.form['study-time'])
        fail_number = int(request.form['fail-number'])
        schoolsup = request.form['schoolsup']
        activities = request.form['activities']
        higher = request.form['higher']
        internet = request.form['internet']
        free_time = int(request.form['free-time'])

        # Prepare the data for prediction
        student_data = {
            "name": name,
            "sex": sex,
            "age": age,
            "study_time": study_time,
            "fail_number": fail_number,
            "schoolsup": schoolsup,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "free_time": free_time
        }

        # Insert data into MongoDB collection
        collection.insert_one(student_data)

        # Prepare the data for the model (make sure the data is formatted correctly)
        input_data = {
            "gender_male": 1 if sex == 'male' else 0,
            "study_time": study_time,
            "fail_number": fail_number,
            "schoolsup_yes": 1 if schoolsup == 'yes' else 0,
            "higher_yes": 1 if higher == 'yes' else 0,
            "internet_yes": 1 if internet == 'yes' else 0,
            "free_time": free_time
        }

        # Convert input_data to DataFrame and apply feature scaling (if necessary)
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model_pipeline.predict(input_df)

        # Convert prediction results to a human-readable format
        result = {
            "G1": prediction[0][0],
            "G2": prediction[0][1],
            "G3": prediction[0][2],
            "Absences": prediction[0][3],
            "Schoolsup": "Yes" if prediction[0][4] == 1 else "No",
            "Internet": "Yes" if prediction[0][5] == 1 else "No",
            "Free Time": prediction[0][6],
            "Health": prediction[0][7]
        }

        return jsonify({"result": result, "student_data": student_data})

if __name__ == '__main__':
    app.run(debug=True)
'''



'''from flask import Flask, render_template, request, jsonify
import pandas as pd
import pymongo
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["project"]
collection = db["future"]

# Load the trained machine learning model
model_pipeline = joblib.load('model_pipeline.pkl')

# Route for the index page
@app.route('/')
def index():
    with open('index.html') as file:
        return file.read()

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form data
        name = request.form.get('name', 'Unknown')
        sex = request.form.get('sex', 'F')
        age = request.form.get('age', '0')
        study_time = request.form.get('studytime', '0')
        fail_number = request.form.get('fail-number', '0')
        activities = request.form.get('activities', '')
        higher = request.form.get('higher', '0')
        internet = request.form.get('internet', '0')
        free_time = request.form.get('freetime', '0')
        famsup = request.form.get('famsup', '0')
        Mjob = request.form.get('Mjob', 'other')
        Fjob = request.form.get('Fjob', 'other')

        # Ensure that numeric fields are integers
        try:
            age = int(age)
            study_time = int(study_time)
            fail_number = int(fail_number)
            higher = int(higher)
            internet = int(internet)
            free_time = int(free_time)
            famsup = int(famsup)
        except ValueError:
            return jsonify({"error": "Invalid input for numeric fields."})

        # Map job categories to integers (or some other suitable mapping)
        Mjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }
        Fjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }

        # Prepare input data (ensure no missing values or invalid types)
        input_data = {
            "sex": 1 if sex == 'M' else 0,
            "age": age,
            "studytime": study_time,
            "failures": fail_number,
            "higher": higher,
            "internet": internet,
            "freetime": free_time,
            "activities": 1 if activities else 0,
            "famsup": famsup,
            "Mjob": Mjob_map.get(Mjob, 0),
            "Fjob": Fjob_map.get(Fjob, 0)
        }

        # Insert data into MongoDB collection
        collection.insert_one(input_data)

        # Prepare data for the model (exclude 'absences' from input features)
        input_df = pd.DataFrame([input_data])

        # Ensure all expected columns are present (using 'preprocessor' to get feature names)
        required_columns = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Add missing columns if any
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default values

        # Preprocess the input data
        try:
            preprocessed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        except ValueError as e:
            return jsonify({"error": f"Preprocessing error: {e}"})

        # Make prediction with the preprocessed data
        try:
            prediction = model_pipeline.named_steps['regressor'].predict(preprocessed_input)
        except Exception as e:
            return jsonify({"error": f"Prediction error: {e}"})

        # Format the prediction result
        result = {
            "Predicted G1": prediction[0][0],
            "Predicted G2": prediction[0][1],
            "Predicted G3": prediction[0][2],
            "Predicted Absences": prediction[0][3]  # 'Absences' is part of target, not input
        }

        return jsonify({"result": result, "student_data": input_data})

if __name__ == '__main__':
    app.run(debug=True)'''
    
    
    
    
    
    
from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
db = client['project']
collection = db['future']

# Function to clean the dataset by removing specified fields
def clean_dataset():
    # Fields to remove from the dataset
    fields_to_remove = [
        'address', 'Pstatus', 'reason', 'nursery', 'r omantic', 'famrel', 
        'Dalc', 'Walc', 'guardian', 'famsize', 'Medu', 'Fedu', 'guardian', 
        'traveltime', 'paid', 'goout', 'health', 'school', 'schoolsup', 'famsup'
    ]
    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)
    print(f"Fields removed from {result.modified_count} documents.")
    # Removing duplicate documents
    pipeline = [
        {"$group": {
            "_id": "$_id",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}}
    ]
    duplicates = collection.aggregate(pipeline)
    for doc in duplicates:
        ids_to_delete = doc['ids'][1:]
        collection.delete_many({"_id": {"$in": ids_to_delete}})
    print("Duplicate documents removed, if any.")

# Route for the main page
@app.route('/')
def index():
    with open('index.html') as file:
        return file.read()

@app.route('/submit', methods=['POST'])
def submit():
    # Collect form data
    data = {
        'name': request.form['name'],
        'sex': request.form['sex'],
        'age': int(request.form['age']),
        'Mjob': request.form['Mjob'],
        'Fjob': request.form['Fjob'],
        'studytime': int(request.form['studytime']),
        'failures': int(request.form['failures']),
        'activities': request.form['activities'],
        'higher': request.form['higher'],
        'internet': request.form['internet'],
        'freetime': int(request.form['freetime']),
        'absences': int(request.form['absences']),
        'G1': int(request.form['G1']),
        'G2': int(request.form['G2']),
        'G3': int(request.form['G3'])
    }
    
    # Insert data into the MongoDB collection
    result = collection.insert_one(data)
    data['_id'] = result.inserted_id
    
    # Clean the dataset
    clean_dataset()
    
    # Load the dataset
    df = pd.DataFrame(list(collection.find()))
    
    # Create a new column 'success_rate'
    df['success_rate'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    
    # Apply condition on absences
    df = df[df['absences'] < 5]
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Define features and target variable
    X = df[['G1', 'G2', 'G3', 'absences']]
    y = df['success_rate']
    
    # Handle NaN values in the target variable
    y = y.dropna()
    X = X.loc[y.index]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions for the latest entry
    latest_entry = pd.DataFrame([data])
    latest_entry = pd.get_dummies(latest_entry, drop_first=True).reindex(columns=X.columns, fill_value=0)
    latest_prediction = model.predict(latest_entry)[0]
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return f"Prediction complete! MSE: {mse}, R²: {r2}\n\nPredicted Success Rate for the latest entry: {latest_prediction}"

if __name__ == "__main__":
    app.run(debug=True)
  
    
    
    



'''from flask import Flask, render_template, request, jsonify
import pandas as pd
import pymongo
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["project"]
collection = db["future"]

# Load the trained machine learning model
model_pipeline = joblib.load('model_pipeline.pkl')

# Route for the index page
@app.route('/')
def index():
    with open('index.html') as file:
        return file.read()

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form data
        name = request.form.get('name', 'Unknown')
        sex = request.form.get('sex', 'F')
        age = request.form.get('age', '0')
        study_time = request.form.get('studytime', '0')
        fail_number = request.form.get('fail-number', '0')
        activities = request.form.get('activities', '')
        higher = request.form.get('higher', '0')
        internet = request.form.get('internet', '0')
        free_time = request.form.get('freetime', '0')
        famsup = request.form.get('famsup', '0')
        Mjob = request.form.get('Mjob', 'other')
        Fjob = request.form.get('Fjob', 'other')

        # Ensure that numeric fields are integers and handle missing data
        try:
            age = int(age) if age else 0
            study_time = int(study_time) if study_time else 0
            fail_number = int(fail_number) if fail_number else 0
            higher = int(higher) if higher else 0
            internet = int(internet) if internet else 0
            free_time = int(free_time) if free_time else 0
            famsup = int(famsup) if famsup else 0
        except ValueError:
            return jsonify({"error": "Invalid input for numeric fields."})

        # Map job categories to integers (or some other suitable mapping)
        Mjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }
        Fjob_map = {
            'teacher': 1,
            'health': 2,
            'services': 3,
            'at_home': 4,
            'other': 5
        }

        # Prepare input data (ensure no missing values or invalid types)
        input_data = {
            "sex": 1 if sex == 'M' else 0,
            "age": age,
            "studytime": study_time,
            "failures": fail_number,
            "higher": higher,
            "internet": internet,
            "freetime": free_time,
            "activities": 1 if activities else 0,
            "famsup": famsup,
            "Mjob": Mjob_map.get(Mjob, 0),
            "Fjob": Fjob_map.get(Fjob, 0),
        }

        # Insert data into MongoDB collection
        collection.insert_one(input_data)

        # Prepare data for the model (exclude 'absences' from input features)
        input_df = pd.DataFrame([input_data])

        # Ensure all expected columns are present (using 'preprocessor' to get feature names)
        required_columns = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Add missing columns if any
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default values

        # Convert all values to numeric (handle NaN / None issues)
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Preprocess the input data (to match model's expectations)
        try:
            preprocessed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        except ValueError as e:
            return jsonify({"error": f"Preprocessing error: {e}"})

        # Make prediction with the preprocessed data
        try:
            prediction = model_pipeline.named_steps['regressor'].predict(preprocessed_input)
        except Exception as e:
            return jsonify({"error": f"Prediction error: {e}"})

        # Format the prediction result
        result = {
            "Predicted G1": prediction[0][0],
            "Predicted G2": prediction[0][1],
            "Predicted G3": prediction[0][2],
            "Predicted Absences": prediction[0][3]  # 'Absences' is part of target, not input
        }

        return jsonify({"result": result, "student_data": input_data})

if __name__ == '__main__':
    app.run(debug=True)
'''