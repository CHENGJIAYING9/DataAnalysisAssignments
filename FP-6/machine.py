import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def run_ml(df):
    X = df[['studytime', 'absences', 'health', 'medu', 'fedu']]
    y = df['grade']

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    sorted_idx = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sorted_idx]
    y_pred_lr_sorted = y_pred_lr[sorted_idx]

    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    sorted_idx = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sorted_idx]
    y_pred_rf_sorted = y_pred_rf[sorted_idx]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Study Time vs Grade
    axs[0].scatter(df['studytime'], df['grade'], alpha=0.6)
    axs[0].set_xlabel('Study Time')
    axs[0].set_ylabel('Grade')
    axs[0].set_title('1. Study Time vs Grade\n')

    # 2. Actual vs Predicted (Linear Regression)
    axs[1].scatter(range(len(y_test_sorted)), y_test_sorted, color='blue', label='Actual', alpha=0.6)
    axs[1].scatter(range(len(y_pred_lr_sorted)), y_pred_lr_sorted, color='red', label='Predicted', alpha=0.6)
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Grade')
    axs[1].set_title('Actual vs Predicted (Linear Regression')
    axs[1].legend()

    # 3. Actual vs Predicted (Random Forest)
    axs[2].scatter(range(len(y_test_sorted)), y_test_sorted, color='blue', label='Actual', alpha=0.6)
    axs[2].scatter(range(len(y_pred_rf_sorted)), y_pred_rf_sorted, color='red', label='Predicted', alpha=0.6)
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Grade')
    axs[2].set_title('Actual vs Predicted (Random Forest)')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    return {
        "LinearRegression": {
            "R2": lr_r2,
            "RMSE": lr_rmse,
            "coef": pd.Series(lr.coef_, index=X.columns)
        },
        "RandomForest": {
            "R2": rf_r2,
            "RMSE": rf_rmse,
            "importance": pd.Series(
                rf.feature_importances_, index=X.columns
            )
        }
    }
