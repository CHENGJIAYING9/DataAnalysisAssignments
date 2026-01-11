import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def _prep_X(df, clip_q=0.95):
    # Winsorize absences by clipping at the upper quantile (default 95%)
    df = df.copy()

    cap = df["absences"].quantile(clip_q)
    df["absences_clip"] = df["absences"].clip(upper=cap)

    X = df[['studytime', 'absences_clip', 'health', 'medu', 'fedu']]
    y = df['grade']
    return X, y


def _fit_rf_train_test(X, y, test_size=0.30, random_state=42, n_estimators=300):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    return {
        "R2": float(r2_score(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "importance": pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
        "y_test": y_test,
        "y_pred": y_pred
    }


def run_improved_all(df, clip_q=0.95):
    X, y = _prep_X(df, clip_q=clip_q)
    return _fit_rf_train_test(X, y)


def run_improved_by_grade(df, threshold=10, clip_q=0.95):
    results = {}

    low_df = df[df['grade'] <= threshold]
    high_df = df[df['grade'] > threshold]

    if len(low_df) > 10:
        Xl, yl = _prep_X(low_df, clip_q=clip_q)
        results['Low-grade'] = _fit_rf_train_test(Xl, yl)

    if len(high_df) > 10:
        Xh, yh = _prep_X(high_df, clip_q=clip_q)
        results['High-grade'] = _fit_rf_train_test(Xh, yh)

    return results
