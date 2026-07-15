"""
Heart Disease Prediction - Model Training Pipeline
====================================================

This script trains and evaluates multiple machine learning classifiers
on the Cleveland Heart Disease dataset, performs hyperparameter tuning
on a Random Forest classifier via GridSearchCV, and persists the final
model, scaler, and evaluation artifacts (plots) to disk.

Usage:
    python train.py

Expected input:
    dataset/Heart_disease_cleveland_new.csv

Outputs:
    models/random_forest.pkl
    models/scaler.pkl
    images/algorithm_comparison.png
    images/confusion_matrix.png
    images/roc_curve.png
    images/feature_importance.png
    images/correlation_heatmap.png
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATASET_PATH = "dataset/Heart_disease_cleveland_new.csv"
MODELS_DIR = "models"
IMAGES_DIR = "images"


def create_output_directories():
    """Create the directories used to store models and generated plots."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------
def load_data(path):
    """Load the heart disease dataset from a CSV file."""
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print(df.shape)
    df.info()
    print(df.describe())
    print(df.isnull().sum())
    return df


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def preprocess_data(df):
    """
    Split features/target, impute missing values, split into train/test
    sets, and scale features using StandardScaler.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    print(df.columns)

    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ----------------------------------------------------------------------------
# Algorithm Comparison
# ----------------------------------------------------------------------------
def compare_algorithms(X_train, X_test, y_train, y_test):
    """Train and compare multiple classification algorithms."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        results.append([name, accuracy])

    results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy"])
    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    print("\n===== Algorithm Comparison =====")
    print(results_df.to_string(index=False))

    return results_df


def plot_algorithm_comparison(results_df):
    """Generate and save a bar plot comparing algorithm accuracies."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Accuracy",
        y="Algorithm",
        data=results_df.sort_values(by="Accuracy", ascending=False),
    )
    plt.title("Algorithm Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Algorithm")
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGES_DIR, "algorithm_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ----------------------------------------------------------------------------
# Hyperparameter Tuning - Random Forest
# ----------------------------------------------------------------------------
def tune_random_forest(X_train, y_train):
    """Perform GridSearchCV hyperparameter tuning on a Random Forest model."""
    parameters = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    grid = GridSearchCV(
        rf,
        parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n===== Random Forest - Best Parameters =====")
    print(grid.best_params_)

    return best_model, grid.best_params_


def update_comparison_with_tuned_rf(results_df, best_model, X_test, y_test):
    """Update the algorithm comparison table with the tuned RF accuracy."""
    rf_pred = best_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    results_df.loc[
        results_df["Algorithm"] == "Random Forest", "Accuracy"
    ] = rf_accuracy

    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    print("\n===== Updated Algorithm Comparison (Tuned Random Forest) =====")
    print(results_df.to_string(index=False))

    return results_df


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_model(best_model, X_test, y_test):
    """Evaluate the final model and print all relevant metrics."""
    y_pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, prob)

    print("\n===== Final Model Evaluation =====")
    print("Accuracy =", accuracy * 100)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1-score =", f1)
    print("ROC-AUC:", roc_auc)

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, prob


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred):
    """Generate and save the confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(IMAGES_DIR, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_roc_curve(best_model, X_test, y_test):
    """Generate and save the ROC curve."""
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(
        os.path.join(IMAGES_DIR, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_importance(best_model, df):
    """Generate and save the feature importance bar plot."""
    feature_importances = best_model.feature_importances_
    feature_names = df.drop("target", axis=1).columns

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    print("\n===== Feature Importance (Top 5) =====")
    print(importance_df.head().to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGES_DIR, "feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return importance_df


def plot_correlation_heatmap(df):
    """Generate and save the correlation heatmap of all numeric features."""
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Features")
    plt.savefig(
        os.path.join(IMAGES_DIR, "correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def save_artifacts(best_model, scaler):
    """Persist the trained model and scaler to disk."""
    joblib.dump(best_model, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("\nModel Saved Successfully")


# ----------------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------------
def main():
    """Run the full training pipeline end to end."""
    create_output_directories()

    # 1. Load data
    df = load_data(DATASET_PATH)

    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 3. Compare baseline algorithms
    results_df = compare_algorithms(X_train, X_test, y_train, y_test)

    # 4. Hyperparameter tuning for Random Forest
    best_model, best_params = tune_random_forest(X_train, y_train)

    # 5. Update comparison table with tuned Random Forest accuracy
    results_df = update_comparison_with_tuned_rf(
        results_df, best_model, X_test, y_test
    )
    plot_algorithm_comparison(results_df)

    # 6. Evaluate the final tuned model
    y_pred, prob = evaluate_model(best_model, X_test, y_test)

    # 7. Generate visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(best_model, X_test, y_test)
    plot_feature_importance(best_model, df)
    plot_correlation_heatmap(df)

    # 8. Save trained model and scaler
    save_artifacts(best_model, scaler)


if __name__ == "__main__":
    main()
