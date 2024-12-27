import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Read the dataset
data = pd.read_csv("DATA-30-ELONG.csv")

# Separate features and target
X = data.drop(columns=['Success'])
y = data['Success']

# Feature selection using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Select the most important features
selected_features = X.columns[clf.feature_importances_.argsort()[::-1][:5]]  # Change 5 to the number of top features you want to select

# Create a new dataset with only the selected features and the target feature
new_data = data[selected_features.tolist() + ['Success']]

# Write the new dataset to a file
new_data.to_csv("Important.csv", index=False)
