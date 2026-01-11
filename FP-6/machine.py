import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


FEATURES = ['studytime', 'absences', 'health', 'medu', 'fedu']


def _train_rf(X, y, random_state=42):
    # Train / test split + Random Forest regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    return {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "importance": pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
    }


def run_ml_all(df):
    # ML analysis on ALL students
    X = df[FEATURES]
    y = df['grade']

    return _train_rf(X, y)


def run_ml_by_grade(df, threshold=10):
    # ML analysis separated into low-grade and high-grade students
    low_df = df[df['grade'] <= threshold]
    high_df = df[df['grade'] > threshold]

    results = {}

    if len(low_df) > 10:
        results['Low-grade'] = _train_rf(
            low_df[FEATURES],
            low_df['grade']
        )

    if len(high_df) > 10:
        results['High-grade'] = _train_rf(
            high_df[FEATURES],
            high_df['grade']
        )

    return results
