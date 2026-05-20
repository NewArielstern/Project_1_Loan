import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

train = pd.read_csv("train-LoanProj.csv")
print(train.columns)

X = train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Married','Credit_History']]
y = train['Loan_Status'].map({'Y':1, 'N':0})

print('Some y None: ',y.isna().sum())

numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
categorical_features = ['Married','Credit_History']

numeric_pipeline = Pipeline([           #pipe for std scale of numeric_features
    ('imputer', SimpleImputer(strategy='mean')),         #nan replacement strategy - mean
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([    #pipe for OnHotEncode scale of categorical_features
    ('imputer', SimpleImputer(strategy='most_frequent')),      #nan replacement strategy - most frequent
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([      #preprocessor for pipe model numeric_features and categorical_features each by their pipe scaler
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

pipe_loan_model = Pipeline([
    ('preprocess', preprocessor),
    ('model', SVC(kernel='rbf', probability=True))
])

pipe_loan_model.fit(X, y)          #fit the x after the preprocessor of the pipe model

feature_names = pipe_loan_model.named_steps['preprocess'].get_feature_names_out()

print(feature_names)

scores = cross_val_score(pipe_loan_model,X,y,cv=5,scoring='accuracy') # cross validation scoring

print(scores)
print(scores.mean())

# Save the model to disk
joblib.dump(pipe_loan_model, 'loan_svm_model.joblib')
print("Model saved successfully!")


