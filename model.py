import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
file_path = 'Student data.csv'
column_names = [
    'First Term GPA', 'Second Term GPA', 'First Language', 'Funding',
    'School', 'FastTrack', 'Coop', 'Residency', 'Gender',
    'Previous Education', 'Age Group', 'High School Average Mark',
    'Math Score', 'English Grade', 'FirstYearPersistence'
]

df = pd.read_csv(file_path, skiprows=20, names=column_names).dropna(how='all')
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert data types
df = df.apply(pd.to_numeric, errors='coerce')

# the last column 'FirstYearPersistence' is the target variable
X = df.drop('FirstYearPersistence', axis=1)
y = df['FirstYearPersistence']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
