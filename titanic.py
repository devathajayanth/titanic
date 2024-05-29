from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the Titanic dataset
dataset_path = "C:/Users/Dell/Downloads/Titanic.csv"
df = pd.read_csv(dataset_path)

# Preprocessing
num_cols = ['age', 'sibsp', 'parch', 'fare']
cat_cols = ['sex', 'embarked', 'class', 'who', 'alone']
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df = pd.get_dummies(df, columns=cat_cols)

# Split features and target variable
X = df.drop(columns=['survived'])
y = df['survived']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Define feature names
feature_names = X.columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input values from the form
        sex = request.form['sex']
        age = float(request.form['age'])
        siblings = int(request.form['siblings'])
        parent_children = int(request.form['parent_children'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        pclass = request.form['class']
        who = request.form['who']
        alone = request.form['alone']

        # Prepare input data for prediction
        input_data = {'sex_female': [1 if sex == 'female' else 0],
                      'sex_male': [1 if sex == 'male' else 0],
                      'age': [age],
                      'sibsp': [siblings],
                      'parch': [parent_children],
                      'fare': [fare],
                      'embarked_C': [1 if embarked == 'C' else 0],
                      'embarked_Q': [1 if embarked == 'Q' else 0],
                      'embarked_S': [1 if embarked == 'S' else 0],
                      'class_First': [1 if pclass == 'First' else 0],
                      'class_Second': [1 if pclass == 'Second' else 0],
                      'class_Third': [1 if pclass == 'Third' else 0],
                      'who_child': [1 if who == 'child' else 0],
                      'who_man': [1 if who == 'man' else 0],
                      'who_woman': [1 if who == 'woman' else 0],
                      'alone_False': [1 if alone == 'no' else 0],
                      'alone_True': [1 if alone == 'yes' else 0]}

        input_df = pd.DataFrame(input_data)

        # Reorder input DataFrame columns to match the order of features in X_train
        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]  # Predicted class (0 or 1)

        return jsonify(prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
