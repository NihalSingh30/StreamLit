import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'cmu-sleep.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'gpa_model.pkl')
META_PATH  = os.path.join(os.path.dirname(__file__), 'model', 'meta.pkl')

FEATURES = [
    'TotalSleepTime',
    'midpoint_sleep',
    'bedtime_mssd',
    'daytime_sleep',
    'frac_nights_with_data',
    'cum_gpa',
    'demo_gender',
    'demo_firstgen',
]

def train_and_save():
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    df = df.replace(r'^\s*$', np.nan, regex=True)

    numeric_cols = FEATURES + ['term_gpa']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['term_gpa', 'TotalSleepTime', 'midpoint_sleep', 'cum_gpa'])

    X = df[FEATURES]
    y = df['term_gpa']

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    cv_r2  = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    cv_mae = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    joblib.dump({
        'features': FEATURES,
        'mae': mae,
        'r2': r2,
        'cv_r2_mean': cv_r2.mean(),
        'cv_mae_mean': (-cv_mae).mean(),
        'n_samples': len(df),
    }, META_PATH)

    return pipe, mae, r2


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        return joblib.load(MODEL_PATH), joblib.load(META_PATH)
    pipe, mae, r2 = train_and_save()
    meta = joblib.load(META_PATH)
    return pipe, meta


if __name__ == '__main__':
    print("Training model...")
    pipe, mae, r2 = train_and_save()
    print(f"Done. MAE={mae:.4f}  R²={r2:.4f}")
