import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # --- your feature logic ---
        df['TenureCategory'] = pd.cut(df['YearsAtCompany'],
                                      bins=[-1, 3, 10, np.inf],
                                      labels=['<3 yrs', '4–10 yrs', '10+ yrs'])

        df['TenureRatio'] = (df['YearsInCurrentRole'] + 1e-5) / (df['YearsAtCompany'] + 1e-5)
        df['TenureRatio'] = df['TenureRatio'].astype(float)

        df['TenureGap'] = (df['YearsAtCompany'] - df['YearsInCurrentRole']).astype(float)

        df['ZeroCompanyTenureFlag'] = ((df['YearsAtCompany'] == 0) & (df['TotalWorkingYears'] > 0)).astype(int)
        df['NewJoinerFlag'] = ((df['YearsAtCompany'] < 2) & (df['TotalWorkingYears'] > 5)).astype(int)

        df['JobRole_Simplified'] = df['JobRole'].apply(
            lambda x: 'Technical' if x in [
                'Healthcare Representative', 'Research Scientist',
                'Laboratory Technician'
            ] else 'Other'
        )

        df['OverTime_JobLevel'] = df['OverTime'].astype(str) + "_" + df['JobLevel'].astype(str)
        df['Travel_Occupation'] = df['BusinessTravel'].astype(str) + "_" + df['JobRole'].astype(str)

        df['SatisfactionMean'] = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
        df['SatisfactionRange'] = (
            df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].max(axis=1)
            - df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].min(axis=1)
        )
        df['SatisfactionStability'] = (df['SatisfactionRange'] == 0).astype(int)

        df['Log_MonthlyIncome'] = np.log1p(df['MonthlyIncome'])
        df['Log_DistanceFromHome'] = np.log1p(df['DistanceFromHome'])

        df['LowIncomeFlag'] = (df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.25)).astype(int)

        df['PromotionPerYear'] = (df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1e-5)).astype(float)

        df['YearsCompany_Satisfaction'] = df['YearsAtCompany'] * df['JobSatisfaction']

        df['StressRisk'] = (
            (df['OverTime'] == 'Yes') &
            (df['JobSatisfaction'] <= 2) &
            (df['SatisfactionMean'] < 2.5)
        ).astype(int)

        # --- store feature names for export ---
        self.feature_names_out_ = df.columns.to_list()

        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
