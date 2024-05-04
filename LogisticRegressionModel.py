from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

data = pd.read_csv("Application_Data.csv")

# Taking away the ID and Status column from the features.
features = data.drop(columns=["Status", "Applicant_ID"])
# Only using the Status column for the labels
labels = data["Status"]

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.6)

# Only using the strings for the features. 
numerical_features = x_train.select_dtypes(include=['number'])
categorical_features = x_train.select_dtypes(include=['object'])

encoder = OneHotEncoder(drop='first', sparse_output=False, categories='auto')

x_train_encoded = encoder.fit_transform(categorical_features)

# Transform the categorical features in the testing set
x_test_encoded = encoder.transform(x_test.select_dtypes(include=['object']))

# Combine encoded categorical features with numerical features
x_train_combined = pd.concat([pd.DataFrame(x_train_encoded, columns=encoder.get_feature_names_out(categorical_features.columns)), numerical_features.reset_index(drop=True)], axis=1)
x_test_combined = pd.concat([pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(categorical_features.columns)), x_test.select_dtypes(include=['number']).reset_index(drop=True)], axis=1)

model = LogisticRegression(max_iter=30000)

# Model fitting and prediction
model.fit(x_train_combined, y_train)
pred = model.predict(x_test_combined)

# Model evaluation metrics
mae = mean_absolute_error(y_test, pred)
f1 = f1_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
print('Mean Absolute Error (MAE): ' + str(mae))
print('F1 Score: ' + str(f1))
print('Accuracy is: ' + str(accuracy))

# Encoded data to csv for visualization
df = pd.DataFrame(x_test_combined)
df.to_csv('testFeaturesEncoded.csv', index=False)

# Finding the most important features.
names = x_train_combined.columns.tolist()
coef = np.abs(model.coef_[0])
sorted_index = np.argsort(coef)[::-1]
print("Feature importance")
for i in sorted_index:
    print(f"{names[i]}: {coef[i]}")