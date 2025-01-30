#!/usr/bin/env python

"""
Single-Class XGBoost Fraud Detection Pipeline
Author: Your Name
Description:
  1) Load the data from CSV
  2) Split into Train / Val / Test
  3) Preprocess with ColumnTransformer (scaling, OHE, target encoding)
  4) Apply SMOTE on the Training set
  5) Single final XGBoost model with GridSearchCV
  6) Evaluate on Train/Val/Test sets
  7) Plot the learning curve of the final model
Usage:
  python single_xgboost_pipeline.py
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn and related
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# XGBoost
import xgboost as xgb

# Target Encoding
from category_encoders import TargetEncoder

# SMOTE
from imblearn.over_sampling import SMOTE

# TQDM for progress bar in GridSearch
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

sns.set_style("whitegrid")


class CreditCardFraudPipeline:
    """
    A single-class pipeline that:
      1) Loads data
      2) Splits into Train/Val/Test
      3) Builds preprocessor (scaling + encoding)
      4) SMOTE on Train
      5) GridSearch for final XGBoost
      6) Evaluates on Train/Val/Test
      7) Plots learning curve
    """
    def __init__(
        self,
        data_path: str = "updatedcreditcard.csv",
        target_col: str = "is_fraud",
        numeric_features: list = None,
        cat_small: list = None,
        cat_high: list = None,
        test_size: float = 0.2,
        val_size: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize the pipeline with default or user-defined settings.
        """
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Default columns
        self.numeric_features = numeric_features or [
            "amt", "distance_km", "age", "city_pop",
            "year", "month", "hour", "day_of_week"
        ]
        self.cat_small = cat_small or ["gender"]  # one-hot
        self.cat_high = cat_high or ["city", "state", "job", "category"]  # target-encode

        # Will hold DataFrames after split
        self.X_train_temp = None
        self.X_val = None
        self.X_test = None
        self.y_train_temp = None
        self.y_val = None
        self.y_test = None

        # Encoded / SMOTE
        self.X_train_sm = None
        self.y_train_sm = None
        self.X_val_enc = None
        self.X_test_enc = None

        self.preprocessor = None
        self.final_feature_names = None

        # Final model from Grid Search
        self.best_estimator = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV data into a DataFrame.
        """
        df = pd.read_csv(self.data_path)
        print(f"[INFO] Data loaded from {self.data_path}. Shape={df.shape}")
        return df

    def split_data(self, df: pd.DataFrame):
        """
        Splits the DataFrame into Train, Validation, and Test sets.
        """
        # Columns needed for features
        feature_cols = self.numeric_features + self.cat_small + self.cat_high

        # Drop any rows missing required columns or target
        df.dropna(subset=feature_cols + [self.target_col], inplace=True)

        X = df[feature_cols]
        y = df[self.target_col].astype(int)

        print("\n[INFO] Splitting data...")
        print("Initial shape of X:", X.shape)
        print("Initial shape of y:", y.shape)
        print("Class distribution:\n", y.value_counts(normalize=True) * 100, "%")

        # First: Train (80%) vs Temp (20%)
        self.X_train_temp, X_temp, self.y_train_temp, y_temp = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Then: from Temp, half for Val (10%), half for Test (10%)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        # Print shapes
        print("\n=== FINAL SPLITS ===")
        print("Train shape:", self.X_train_temp.shape, self.y_train_temp.shape)
        print("Validation shape:", self.X_val.shape, self.y_val.shape)
        print("Test shape:", self.X_test.shape, self.y_test.shape)

        print("\nTrain class distribution:\n", self.y_train_temp.value_counts(normalize=True)*100, "%")
        print("Val class distribution:\n", self.y_val.value_counts(normalize=True)*100, "%")
        print("Test class distribution:\n", self.y_test.value_counts(normalize=True)*100, "%")

    def build_preprocessor(self):
        """
        Creates and returns the ColumnTransformer with scaling, OHE, and target encoding.
        """
        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        ohe_transformer = Pipeline([
            ("ohe", OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        target_encoder = Pipeline([
            ("target_enc", TargetEncoder(smoothing=0.3))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("ohe", ohe_transformer, self.cat_small),
                ("te",  target_encoder,  self.cat_high),
            ],
            remainder="drop"
        )

    def encode_and_smote(self):
        """
        Fits the preprocessor on the training data, applies SMOTE, and encodes Val/Test.
        """
        # Fit on train only
        X_train_enc = self.preprocessor.fit_transform(self.X_train_temp, self.y_train_temp)
        print("\n[INFO] Shape of X_train_enc BEFORE SMOTE:", X_train_enc.shape)
        print("Class distribution in y_train_temp BEFORE SMOTE:", np.bincount(self.y_train_temp))

        # SMOTE
        sm = SMOTE(random_state=self.random_state)
        self.X_train_sm, self.y_train_sm = sm.fit_resample(X_train_enc, self.y_train_temp)

        print("\n[INFO] Shape of X_train_enc AFTER SMOTE:", self.X_train_sm.shape)
        print("Class distribution in y_train_sm AFTER SMOTE:", np.bincount(self.y_train_sm))

        # Encode validation & test (do not fit again)
        self.X_val_enc  = self.preprocessor.transform(self.X_val)
        self.X_test_enc = self.preprocessor.transform(self.X_test)

        # Build final feature name list
        ohe_step = self.preprocessor.named_transformers_['ohe'].named_steps['ohe']
        ohe_feature_names = ohe_step.get_feature_names_out(self.cat_small)
        te_cols = [f"{col}_te" for col in self.cat_high]

        self.final_feature_names = self.numeric_features + list(ohe_feature_names) + te_cols
        print("\n[DEBUG] Final Encoded Feature List (Train/Val/Test):")
        print(self.final_feature_names)

    def run_grid_search(self, param_grid: dict, cv_folds: int = 3, scale_pos_weight: int = 1):
        """
        Performs a single Grid Search for XGBoost on the SMOTE-transformed training data.
        """
        print("\n[DEBUG] Starting Grid Search for XGBoost...")

        # Estimate total combos for the progress bar
        total_combinations = math.prod(len(v) for v in param_grid.values())
        total_iterations = total_combinations * cv_folds

        # TQDM progress bar
        progress_bar = tqdm(total=total_iterations, desc="GridSearchCV Progress", unit="fit")

        with tqdm_joblib(progress_bar):
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                tree_method='hist',
                verbosity=1,
                scale_pos_weight=scale_pos_weight
            )

            gs = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='accuracy',
                cv=cv_folds,
                n_jobs=-1,
                verbose=0
            )
            gs.fit(self.X_train_sm, self.y_train_sm)

        progress_bar.close()

        print("\n[INFO] Best Parameters from Grid Search:\n", gs.best_params_)
        print("\n[INFO] Best Cross-Validation Accuracy:\n", gs.best_score_)

        # Save the best estimator
        self.best_estimator = gs.best_estimator_

    def evaluate_model(self, model, X, y, dataset_name="Train"):
        """
        Evaluates the given model on (X, y) and prints performance metrics.
        """
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"\n=== [XGBoost] MODEL PERFORMANCE ON {dataset_name} SET ===")
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y, preds))
        print("Confusion Matrix:\n", confusion_matrix(y, preds))

    def plot_model_learning_curve(self, model, X, y, cv: int = 5, scoring: str = "accuracy"):
        """
        Plots the learning curve of the final model.
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=cv,
            n_jobs=-1,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            random_state=self.random_state
        )

        train_means = np.mean(train_scores, axis=1)
        train_stds  = np.std(train_scores, axis=1)
        val_means   = np.mean(val_scores, axis=1)
        val_stds    = np.std(val_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_means, 'o-', color='blue', label='Training')
        plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds,
                         alpha=0.1, color='blue')

        plt.plot(train_sizes, val_means, 'o-', color='red', label='Validation')
        plt.fill_between(train_sizes, val_means - val_stds, val_means + val_stds,
                         alpha=0.1, color='red')

        plt.title('Learning Curve: XGBoost')
        plt.xlabel('Training Set Size')
        plt.ylabel(scoring.title())
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def run_pipeline(self):
        """
        Orchestrates the entire process:
          1) Load & split data
          2) Build preprocessor and SMOTE
          3) Grid search for final XGBoost
          4) Evaluate best model on Train, Val, Test
          5) Plot learning curve
        """

        # 1) Load data
        df = self.load_data()

        # 2) Split data
        self.split_data(df)

        # 3) Build preprocessor & encode + SMOTE
        self.build_preprocessor()
        self.encode_and_smote()

        # 4) Single final XGBoost with GridSearch
        param_grid = {
            "n_estimators": [200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [8, 10],
            "subsample": [0.8],
            "colsample_bytree": [0.8, 1.0]
        }
        # Adjust scale_pos_weight if you know the imbalance ratio, e.g. ratio of (non-fraud/fraud).
        # Example: scale_pos_weight=100 means the negative class is ~100 times bigger than positive.
        self.run_grid_search(param_grid=param_grid, cv_folds=3, scale_pos_weight=1)

        # 5) Evaluate on Train, Val, Test with the best estimator
        # Train (with SMOTE data)
        self.evaluate_model(self.best_estimator, self.X_train_sm, self.y_train_sm, "Train (SMOTE)")
        # Validation
        self.evaluate_model(self.best_estimator, self.X_val_enc, self.y_val, "Validation")
        # Test
        self.evaluate_model(self.best_estimator, self.X_test_enc, self.y_test, "Test")

        # 6) Plot Learning Curve (optional)
        self.plot_model_learning_curve(self.best_estimator, self.X_train_sm, self.y_train_sm, cv=3)


def main():
    # Create instance of pipeline
    pipeline = CreditCardFraudPipeline(
        data_path="updatedcreditcard.csv",
        target_col="is_fraud",
        random_state=42
    )
    # Run everything
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()



#deployment - backtesting AUC ROC Guiney score - models discrimination power - how well it separates fraud from non-fraud
