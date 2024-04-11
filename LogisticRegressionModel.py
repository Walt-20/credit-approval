from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

data = pd.read_csv("Application_Data.csv")

features = data.drop(columns=["Status", "Applicant_ID"])
labels = data["Status"]

categorical_features = features.select_dtypes(include=['object'])

x_train, x_test, y_train, y_test = train_test_split(categorical_features, labels, test_size=0.2)

encoder = OneHotEncoder(drop='first', sparse_output=False, categories='auto')

x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

model = LogisticRegression()

model.fit(x_train_encoded, y_train)
pred = model.predict(x_test_encoded)

mae = mean_absolute_error(y_test, pred)
print('Mean Absolute Error (MAE): ' + str(mae))