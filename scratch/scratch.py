import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    'Age': [22, 38, 26, None, 35],
    'Fare': [7.25, 71.83, 7.92, 8.05, 53.1],
    'Sex': ['male', 'female', 'female', 'female', 'male'],
    'Embarked': ['S', 'C', 'S', 'S', 'S'],
    'Survived': [0, 1, 1, 1, 0]
})

X = df.drop(columns='Survived')
y = df['Survived']

numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = Pipeline([
	('imputer', SimpleImputer(strategy='mean')),
	('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
	('num', numeric_transformer, numeric_features),
	('cat', categorical_transformer, categorical_features)
])

preprocessor.set_output(transform='pandas')

pipe = Pipeline([
	('preprocess', preprocessor),
	('model', LogisticRegression(max_iter=1000))
])

param_grid = {
	'model__C': [0.1, 1, 10],
	'model__penalty': ['l2']
}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X, y)

print("Best params:", grid.best_params_)
print("CV score:", grid.best_score_)
print("Final predictions:", grid.predict(X))

