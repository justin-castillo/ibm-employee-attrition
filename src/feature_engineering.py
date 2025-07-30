import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Inherit for use in GridSearch and fit_transform()
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Bin tenure into 3 categories for simplicity 
        df['TenureCategory'] = pd.cut(df['YearsAtCompany'],
                                      bins=[-1, 3, 10, np.inf],
                                      labels=['<=3 yrs', '4–10 yrs', '10< yrs'])

        # 1.0 = employee was in same role the entire time at this company
        # 1e-5: avoid division by 0
        df['TenureRatio'] = (df['YearsInCurrentRole'] + 1e-5) / (df['YearsAtCompany'] + 1e-5)
        df['TenureRatio'] = df['TenureRatio'].astype(float)

        # How much time was spent in previous roles 
        df['TenureGap'] = (df['YearsAtCompany'] - df['YearsInCurrentRole']).astype(float)

        # Newcomers with any prior experience 
        df['ZeroCompanyTenureFlag'] = ((df['YearsAtCompany'] == 0) & (df['TotalWorkingYears'] > 0)).astype(int)

        # Highly experienced newcomers 
        df['NewJoinerFlag'] = ((df['YearsAtCompany'] < 2) & (df['TotalWorkingYears'] > 5)).astype(int)

        # Jobs levels that have overtime or not 
        df['OverTime_JobLevel'] = df['OverTime'].astype(str) + "_" + df['JobLevel'].astype(str)

        # Roles that require travel or not 
        df['Travel_Occupation'] = df['BusinessTravel'].astype(str) + "_" + df['JobRole'].astype(str)

        # Measure overall employee satisfaction 
        df['SatisfactionMean'] = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)

        # Consistency of satisfaction across environment/role/relationship
        df['SatisfactionRange'] = (
            df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].max(axis=1)
            - df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].min(axis=1)
        )

        # Classify an employee's satisfaction scores as consistent or not consistent 
        df['SatisfactionStability'] = (df['SatisfactionRange'] == 0).astype(int)

        # Log-transform MonthlyIncome to reduce skew
        df['Log_MonthlyIncome'] = np.log1p(df['MonthlyIncome'])

        # Log-transform DistanceFromHome to reduce skew
        df['Log_DistanceFromHome'] = np.log1p(df['DistanceFromHome'])

        # Especially low earners
        df['LowIncomeFlag'] = (df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.25)).astype(int)

        # Frequency of promotions
        df['PromotionPerYear'] = (df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1e-5)).astype(float)

        # How tenure affects satisfaction
        df['YearsCompany_Satisfaction'] = df['YearsAtCompany'] * df['JobSatisfaction']

        # Identify high-risk groups
        df['StressRisk'] = (
            (df['OverTime'] == 'Yes') &
            (df['JobSatisfaction'] <= 2) &
            (df['SatisfactionMean'] < 2.5)
        ).astype(int)

        cols_to_drop = [
        'YearsAtCompany', 'YearsInCurrentRole', 'BusinessTravel',
        'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction',
        'MonthlyIncome'
        ]
        df.drop(columns=cols_to_drop, inplace=True)


        # For interpretability 
        self.feature_names_out_ = df.columns.to_list()

        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
