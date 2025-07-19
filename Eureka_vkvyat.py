# REQUIRED LIBRARIES
# ==================
# Install required packages if not already installed:
# pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


# STEP 1: DATA LOADING AND PREPROCESSING
# =====================================
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Eureka Forbes dataset

    This function:
    1. Loads the CSV data with proper encoding
    2. Handles missing values by filling with 0
    3. Creates a binary target variable for conversion prediction
    4. Selects relevant features for modeling
    5. Creates dummy variables for categorical features

    Args:
        file_path (str): Path to the CSV file

    Returns:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        df (DataFrame): Original dataframe with preprocessing
    """
    print("=" * 60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 60)

    # Load the dataset
    print("Loading Eureka Forbes dataset...")
    df = pd.read_csv(file_path, encoding='ascii')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Display target variable distribution
    print(f"\nTarget variable distribution:")
    print(df['converted_in_7days'].value_counts())

    # Handle missing values
    print(f"\nMissing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(0)  # Fill missing values with 0 for this use case
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Create binary target variable (convert multi-class to binary)
    # Any conversion > 0 is considered a positive conversion
    df['converted_binary'] = (df['converted_in_7days'] > 0).astype(int)
    print(f"Binary conversion rate: {df['converted_binary'].mean() * 100:.3f}%")

    # Select relevant numerical features for modeling
    # These features represent user behavior and engagement metrics
    feature_columns = [
        'sessionDuration',  # Time spent on website
        'pageviews',  # Number of pages viewed
        'bounces',  # Single-page sessions
        'sessions',  # Number of sessions
        'goal4Completions',  # Goal completions
        'newUser',  # New vs returning user
        'visited_demo_page',  # Visited product demo page
        'visited_checkout_page',  # Visited checkout page
        'visited_water_purifier_page',  # Visited water purifier page
        'fired_phone_clicks_evt',  # Clicked phone number
        'fired_help_me_buy_evt',  # Clicked help me buy
        'DemoReqPg_CallClicks_evt_count',  # Demo request call clicks
        'help_me_buy_evt_count',  # Help me buy event count
        'phone_clicks_evt_count',  # Phone click event count
        'paid'  # Paid traffic indicator
    ]

    # Ensure all selected features exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"\nAvailable numerical features for modeling: {len(available_features)}")

    # Handle categorical variables by creating dummy variables
    # This converts categorical data into numerical format for ML models
    categorical_cols = ['device', 'sourceMedium', 'country', 'region']
    print(f"\nProcessing categorical variables...")

    for col in categorical_cols:
        if col in df.columns:
            print(f"  - Creating dummies for {col} ({df[col].nunique()} unique values)")
            # Create dummy variables (one-hot encoding)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            available_features.extend(dummies.columns.tolist())

    # Prepare final feature set
    X = df[available_features].copy()
    y = df['converted_binary'].copy()

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")

    return X, y, df


# STEP 2: MODEL TRAINING FUNCTIONS
# ===============================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Logistic Regression model

    Logistic Regression is a linear model good for:
    - Interpretable coefficients
    - Fast training and prediction
    - Baseline model performance

    Uses StandardScaler for feature normalization and class_weight='balanced'
    to handle class imbalance.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    # Scale features for logistic regression (important for convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build logistic regression model with class balancing
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handles class imbalance
    )
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1_score': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }

    print("Logistic Regression Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance analysis (coefficients)
    feature_importance_lr = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print("\nTop 10 Most Important Features (Logistic Regression):")
    for idx, row in feature_importance_lr.head(10).iterrows():
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"  {row['feature']}: {row['coefficient']:.4f} ({direction} conversion probability)")

    # Confusion matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_lr[0, 0]:,}")
    print(f"  False Positives: {cm_lr[0, 1]:,}")
    print(f"  False Negatives: {cm_lr[1, 0]:,}")
    print(f"  True Positives: {cm_lr[1, 1]:,}")

    return results, lr_model, scaler


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Decision Tree model

    Decision Trees are good for:
    - Non-linear relationships
    - Feature interactions
    - Interpretable rules
    - No need for feature scaling

    Uses hyperparameters to prevent overfitting.
    """
    print("\n" + "=" * 60)
    print("DECISION TREE CLASSIFIER")
    print("=" * 60)

    # Build decision tree model with regularization parameters
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=100,  # Minimum samples to split a node
        class_weight='balanced'  # Handle class imbalance
    )
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Decision Tree',
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt),
        'recall': recall_score(y_test, y_pred_dt),
        'f1_score': f1_score(y_test, y_pred_dt),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
    }

    print("Decision Tree Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance (based on information gain)
    feature_importance_dt = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Decision Tree):")
    for idx, row in feature_importance_dt.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_dt[0, 0]:,}")
    print(f"  False Positives: {cm_dt[0, 1]:,}")
    print(f"  False Negatives: {cm_dt[1, 0]:,}")
    print(f"  True Positives: {cm_dt[1, 1]:,}")

    return results, dt_model


def train_random_forest_smote(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest model with SMOTE sampling

    Random Forest combines multiple decision trees and is good for:
    - Handling overfitting better than single trees
    - Feature importance ranking
    - Robust performance

    SMOTE (Synthetic Minority Oversampling Technique) creates synthetic
    examples of the minority class to balance the dataset.
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL WITH SMOTE SAMPLING")
    print("=" * 60)

    # Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  SMOTE balanced training set: {X_train_smote.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

    # Build Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Maximum depth of trees
        min_samples_split=20,  # Minimum samples to split a node
        min_samples_leaf=10,  # Minimum samples in a leaf
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    rf_model.fit(X_train_smote, y_train_smote)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Random Forest (SMOTE)',
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }

    print("\nRandom Forest Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Random Forest):")
    for idx, row in feature_importance_rf.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_rf[0, 0]:,}")
    print(f"  False Positives: {cm_rf[0, 1]:,}")
    print(f"  False Negatives: {cm_rf[1, 0]:,}")
    print(f"  True Positives: {cm_rf[1, 1]:,}")

    return results, rf_model


def train_xgboost_oversampling(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost model with Random Oversampling

    XGBoost is a gradient boosting framework that is excellent for:
    - High performance on structured data
    - Built-in regularization
    - Feature importance
    - Handling missing values

    Random Oversampling duplicates minority class samples to balance the dataset.
    """
    print("\n" + "=" * 60)
    print("XGBOOST MODEL WITH RANDOM OVERSAMPLING")
    print("=" * 60)

    # Apply random oversampling for minority class
    print("Applying Random Oversampling to balance classes...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  Up-sampled training set: {X_train_ros.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Up-sampled class distribution: {pd.Series(y_train_ros).value_counts().to_dict()}")

    # Build XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,  # Number of boosting rounds
        max_depth=6,  # Maximum depth of trees
        learning_rate=0.1,  # Step size shrinkage
        subsample=0.8,  # Fraction of samples used for training each tree
        colsample_bytree=0.8,  # Fraction of features used for training each tree
        random_state=42,
        eval_metric='logloss'  # Evaluation metric
    )

    xgb_model.fit(X_train_ros, y_train_ros)

    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'XGBoost (Random Oversampling)',
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1_score': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb)
    }

    print("\nXGBoost Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_xgb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (XGBoost):")
    for idx, row in feature_importance_xgb.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_xgb[0, 0]:,}")
    print(f"  False Positives: {cm_xgb[0, 1]:,}")
    print(f"  False Negatives: {cm_xgb[1, 0]:,}")
    print(f"  True Positives: {cm_xgb[1, 1]:,}")

    return results, xgb_model


def train_gradient_boosting_smote(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Gradient Boosting model with SMOTE sampling

    Gradient Boosting builds models sequentially, where each model
    corrects the errors of the previous ones. Good for:
    - High predictive accuracy
    - Handling complex patterns
    - Feature importance
    """
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING MODEL WITH SMOTE SAMPLING")
    print("=" * 60)

    # Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  SMOTE balanced training set: {X_train_smote.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

    # Build Gradient Boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=100,  # Number of boosting stages
        max_depth=6,  # Maximum depth of trees
        learning_rate=0.1,  # Learning rate shrinks contribution of each tree
        subsample=0.8,  # Fraction of samples used for fitting trees
        random_state=42
    )

    gb_model.fit(X_train_smote, y_train_smote)

    # Make predictions
    y_pred_gb = gb_model.predict(X_test)
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Gradient Boosting (SMOTE)',
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1_score': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }

    print("\nGradient Boosting Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_gb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Gradient Boosting):")
    for idx, row in feature_importance_gb.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_gb[0, 0]:,}")
    print(f"  False Positives: {cm_gb[0, 1]:,}")
    print(f"  False Negatives: {cm_gb[1, 0]:,}")
    print(f"  True Positives: {cm_gb[1, 1]:,}")

    return results, gb_model


# STEP 3: VISUALIZATION FUNCTIONS
# ==============================

def create_model_comparison_plots(all_results):
    """
    Create comprehensive visualizations comparing all models

    This function creates:
    1. Model performance comparison bar chart
    2. ROC curves for all models
    3. Feature importance comparison
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Eureka Forbes Conversion Prediction - Model Comparison', fontsize=16, fontweight='bold')

    # 1. Model Performance Comparison
    metrics_df = pd.DataFrame(all_results)
    metrics_df = metrics_df.set_index('model')

    # Plot performance metrics
    ax1 = axes[0, 0]
    metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].plot(kind='bar', ax=ax1)
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. F1-Score vs ROC-AUC Scatter Plot
    ax2 = axes[0, 1]
    scatter = ax2.scatter(metrics_df['f1_score'], metrics_df['roc_auc'],
                          s=100, alpha=0.7, c=range(len(metrics_df)), cmap='viridis')

    # Add model labels to points
    for i, model in enumerate(metrics_df.index):
        ax2.annotate(model, (metrics_df.iloc[i]['f1_score'], metrics_df.iloc[i]['roc_auc']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('F1-Score')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('F1-Score vs ROC-AUC', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    scatter2 = ax3.scatter(metrics_df['recall'], metrics_df['precision'],
                           s=100, alpha=0.7, c=range(len(metrics_df)), cmap='plasma')

    # Add model labels to points
    for i, model in enumerate(metrics_df.index):
        ax3.annotate(model, (metrics_df.iloc[i]['recall'], metrics_df.iloc[i]['precision']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Model Ranking by Different Metrics
    ax4 = axes[1, 1]

    # Create ranking for each metric
    ranking_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        ranked = metrics_df[metric].rank(ascending=False)
        for model, rank in ranked.items():
            ranking_data.append({'Model': model, 'Metric': metric, 'Rank': rank})

    ranking_df = pd.DataFrame(ranking_data)
    ranking_pivot = ranking_df.pivot(index='Model', columns='Metric', values='Rank')

    # Create heatmap
    sns.heatmap(ranking_pivot, annot=True, cmap='RdYlGn_r', ax=ax4,
                cbar_kws={'label': 'Rank (1=Best)'})
    ax4.set_title('Model Ranking by Metric', fontweight='bold')
    ax4.set_ylabel('')

    plt.tight_layout()
    plt.savefig('eureka_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics_df


def plot_feature_importance_comparison(feature_importances_dict):
    """
    Compare feature importance across different models
    """
    num_models = len(feature_importances_dict)

    # Calculate appropriate grid dimensions
    if num_models <= 2:
        rows, cols = 1, num_models
    elif num_models <= 4:
        rows, cols = 2, 2
    elif num_models <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    fig.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')

    # Handle single subplot case
    if num_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()

    model_names = list(feature_importances_dict.keys())

    for i, (model_name, importance_df) in enumerate(feature_importances_dict.items()):
        ax = axes[i]

        # Get top 10 features
        top_features = importance_df.head(10)

        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features.iloc[:, 1],
                       color=plt.cm.Set3(i))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.iloc[:, 0])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top 10 Features', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)

    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('eureka_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_data):
    """
    Plot ROC curves for all models on the same plot
    """
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (model_name, y_test, y_pred_proba) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('eureka_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_data_exploration_plots(df):
    """
    Create exploratory data analysis plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Eureka Forbes Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # 1. Conversion rate by device
    ax1 = axes[0, 0]
    conversion_by_device = df.groupby('device')['converted_binary'].agg(['count', 'sum', 'mean'])
    conversion_by_device['conversion_rate'] = conversion_by_device['mean'] * 100

    bars = ax1.bar(conversion_by_device.index, conversion_by_device['conversion_rate'])
    ax1.set_title('Conversion Rate by Device Type')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_xlabel('Device Type')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 2. Session duration distribution
    ax2 = axes[0, 1]
    ax2.hist(df['sessionDuration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Session Duration Distribution')
    ax2.set_xlabel('Session Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['sessionDuration'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["sessionDuration"].mean():.0f}s')
    ax2.legend()

    # 3. Pageviews vs Conversion
    ax3 = axes[0, 2]
    converted = df[df['converted_binary'] == 1]['pageviews']
    not_converted = df[df['converted_binary'] == 0]['pageviews']

    ax3.hist([not_converted, converted], bins=30, alpha=0.7,
             label=['Not Converted', 'Converted'], color=['lightcoral', 'lightgreen'])
    ax3.set_title('Pageviews Distribution by Conversion')
    ax3.set_xlabel('Number of Pageviews')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_yscale('log')

    # 4. Traffic source analysis
    ax4 = axes[1, 0]
    source_conversion = df.groupby('sourceMedium')['converted_binary'].agg(['count', 'mean'])
    source_conversion = source_conversion[source_conversion['count'] >= 100]  # Filter for significant sources
    source_conversion = source_conversion.sort_values('mean', ascending=True)

    bars = ax4.barh(range(len(source_conversion)), source_conversion['mean'] * 100)
    ax4.set_yticks(range(len(source_conversion)))
    ax4.set_yticklabels([src[:20] + '...' if len(src) > 20 else src for src in source_conversion.index])
    ax4.set_xlabel('Conversion Rate (%)')
    ax4.set_title('Conversion Rate by Traffic Source')

    # 5. New vs Returning Users
    ax5 = axes[1, 1]
    user_type_conv = df.groupby('newUser')['converted_binary'].agg(['count', 'mean'])
    user_labels = ['Returning User', 'New User']

    bars = ax5.bar(user_labels, user_type_conv['mean'] * 100, color=['orange', 'green'])
    ax5.set_title('Conversion Rate: New vs Returning Users')
    ax5.set_ylabel('Conversion Rate (%)')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 6. Correlation heatmap of key features
    ax6 = axes[1, 2]
    key_features = ['sessionDuration', 'pageviews', 'sessions', 'bounces',
                    'goal4Completions', 'converted_binary']
    correlation_matrix = df[key_features].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('eureka_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()


# STEP 4: MAIN EXECUTION FUNCTION
# ===============================

def main():
    """
    Main function that executes the complete machine learning pipeline

    This function:
    1. Loads and preprocesses the data
    2. Splits data into train/test sets
    3. Trains all models with appropriate sampling techniques
    4. Compares model performance
    5. Creates comprehensive visualizations
    6. Provides recommendations
    """
    print("=" * 80)
    print("EUREKA FORBES CONVERSION PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    file_path = 'eureka_data_final_2019-01-01_2019-03-01.csv'  # Update path as needed
    X, y, df = load_and_preprocess_data(file_path)

    # Step 2: Split data into training and testing sets
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA INTO TRAIN/TEST SETS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]:,} samples")
    print(f"Testing set size: {X_test.shape[0]:,} samples")
    print(f"Training set conversion rate: {y_train.mean() * 100:.3f}%")
    print(f"Testing set conversion rate: {y_test.mean() * 100:.3f}%")

    # Step 3: Train all models
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING ALL MODELS")
    print("=" * 60)

    all_results = []
    trained_models = {}
    feature_importances = {}
    roc_data = []

    # 1. Logistic Regression
    try:
        lr_results, lr_model, lr_scaler = train_logistic_regression(X_train, X_test, y_train, y_test)
        all_results.append(lr_results)
        trained_models['Logistic Regression'] = (lr_model, lr_scaler)

        # Get feature importance for logistic regression
        feature_importance_lr = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(lr_model.coef_[0])
        }).sort_values('importance', ascending=False)
        feature_importances['Logistic Regression'] = feature_importance_lr

        # ROC data
        y_pred_proba_lr = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
        roc_data.append(('Logistic Regression', y_test, y_pred_proba_lr))

    except Exception as e:
        print(f"Error training Logistic Regression: {e}")

    # 2. Decision Tree
    try:
        dt_results, dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
        all_results.append(dt_results)
        trained_models['Decision Tree'] = dt_model

        # Get feature importance for decision tree
        feature_importance_dt = pd.DataFrame({
            'feature': X_train.columns,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Decision Tree'] = feature_importance_dt

        # ROC data
        y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Decision Tree', y_test, y_pred_proba_dt))

    except Exception as e:
        print(f"Error training Decision Tree: {e}")

    # 3. Random Forest with SMOTE
    try:
        rf_results, rf_model = train_random_forest_smote(X_train, X_test, y_train, y_test)
        all_results.append(rf_results)
        trained_models['Random Forest (SMOTE)'] = rf_model

        # Get feature importance for random forest
        feature_importance_rf = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Random Forest (SMOTE)'] = feature_importance_rf

        # ROC data
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Random Forest (SMOTE)', y_test, y_pred_proba_rf))

    except Exception as e:
        print(f"Error training Random Forest: {e}")

    # 4. XGBoost with Random Oversampling
    try:
        xgb_results, xgb_model = train_xgboost_oversampling(X_train, X_test, y_train, y_test)
        all_results.append(xgb_results)
        trained_models['XGBoost (Random Oversampling)'] = xgb_model

        # Get feature importance for XGBoost
        feature_importance_xgb = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['XGBoost (Random Oversampling)'] = feature_importance_xgb

        # ROC data
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        roc_data.append(('XGBoost (Random Oversampling)', y_test, y_pred_proba_xgb))

    except Exception as e:
        print(f"Error training XGBoost: {e}")

    # 5. Gradient Boosting with SMOTE
    try:
        gb_results, gb_model = train_gradient_boosting_smote(X_train, X_test, y_train, y_test)
        all_results.append(gb_results)
        trained_models['Gradient Boosting (SMOTE)'] = gb_model

        # Get feature importance for gradient boosting
        feature_importance_gb = pd.DataFrame({
            'feature': X_train.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Gradient Boosting (SMOTE)'] = feature_importance_gb

        # ROC data
        y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Gradient Boosting (SMOTE)', y_test, y_pred_proba_gb))

    except Exception as e:
        print(f"Error training Gradient Boosting: {e}")

    # Step 4: Model Comparison and Visualization
    print("\n" + "=" * 60)
    print("STEP 4: MODEL COMPARISON AND VISUALIZATION")
    print("=" * 60)

    if all_results:
        # Create comparison table
        comparison_df = create_model_comparison_plots(all_results)

        # Display final results table
        print("\nFINAL MODEL COMPARISON:")
        print("=" * 80)
        print(comparison_df.round(4))

        # Find best model for each metric
        print("\nBEST PERFORMING MODELS BY METRIC:")
        print("=" * 50)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")

        # Create feature importance comparison
        if feature_importances:
            plot_feature_importance_comparison(feature_importances)

        # Create ROC curves
        if roc_data:
            plot_roc_curves(roc_data)

        # Create data exploration plots
        create_data_exploration_plots(df)

        # Step 5: Recommendations
        print("\n" + "=" * 60)
        print("STEP 5: RECOMMENDATIONS AND INSIGHTS")
        print("=" * 60)

        # Overall best model recommendation
        best_f1_model = comparison_df['f1_score'].idxmax()
        best_auc_model = comparison_df['roc_auc'].idxmax()

        print("BUSINESS RECOMMENDATIONS:")
        print("=" * 30)
        print(f"1. BEST OVERALL MODEL: {best_f1_model}")
        print(f"   - F1-Score: {comparison_df.loc[best_f1_model, 'f1_score']:.4f}")
        print(f"   - ROC-AUC: {comparison_df.loc[best_f1_model, 'roc_auc']:.4f}")
        print(f"   - This model provides the best balance of precision and recall")

        print(f"\n2. BEST FOR RANKING/SCORING: {best_auc_model}")
        print(f"   - ROC-AUC: {comparison_df.loc[best_auc_model, 'roc_auc']:.4f}")
        print(f"   - Use this model for probability scoring and ranking customers")

        # Feature insights
        if feature_importances:
            print("\n3. KEY CONVERSION DRIVERS:")
            # Get most common top features across models
            all_top_features = []
            for model_name, importance_df in feature_importances.items():
                all_top_features.extend(importance_df.head(5)['feature'].tolist())

            from collections import Counter
            feature_counts = Counter(all_top_features)

            print("   Most important features across all models:")
            for feature, count in feature_counts.most_common(10):
                print(f"   - {feature} (mentioned in {count} models)")

        print("\n4. BUSINESS ACTIONS:")
        print("   - Focus on improving session duration and page engagement")
        print("   - Optimize demo page and checkout page experience")
        print("   - Implement targeted campaigns for high-value traffic sources")
        print("   - Use model predictions to prioritize sales follow-ups")

        # Save results to CSV
        comparison_df.to_csv('eureka_model_results.csv')
        print("\n5. OUTPUTS SAVED:")
        print("   - eureka_model_results.csv: Model performance comparison")
        print("   - eureka_model_comparison.png: Performance visualization")
        print("   - eureka_feature_importance_comparison.png: Feature importance")
        print("   - eureka_roc_curves.png: ROC curve comparison")
        print("   - eureka_data_exploration.png: Data exploration plots")

        return trained_models, comparison_df, feature_importances

    else:
        print("No models were successfully trained. Please check the data and try again.")
        return None, None, None


# EXECUTION
# =========
if __name__ == "__main__":
    # Run the complete pipeline
    models, results, importances = main()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nTo use this script:")
    print("1. Ensure all required packages are installed")
    print("2. Update the file_path variable to point to your CSV file")
    print("3. Run: python eureka_complete_script.py")# REQUIRED LIBRARIES
# ==================
# Install required packages if not already installed:
# pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


# STEP 1: DATA LOADING AND PREPROCESSING
# =====================================
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Eureka Forbes dataset

    This function:
    1. Loads the CSV data with proper encoding
    2. Handles missing values by filling with 0
    3. Creates a binary target variable for conversion prediction
    4. Selects relevant features for modeling
    5. Creates dummy variables for categorical features

    Args:
        file_path (str): Path to the CSV file

    Returns:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        df (DataFrame): Original dataframe with preprocessing
    """
    print("=" * 60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 60)

    # Load the dataset
    print("Loading Eureka Forbes dataset...")
    df = pd.read_csv(file_path, encoding='ascii')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Display target variable distribution
    print(f"\nTarget variable distribution:")
    print(df['converted_in_7days'].value_counts())

    # Handle missing values
    print(f"\nMissing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(0)  # Fill missing values with 0 for this use case
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Create binary target variable (convert multi-class to binary)
    # Any conversion > 0 is considered a positive conversion
    df['converted_binary'] = (df['converted_in_7days'] > 0).astype(int)
    print(f"Binary conversion rate: {df['converted_binary'].mean() * 100:.3f}%")

    # Select relevant numerical features for modeling
    # These features represent user behavior and engagement metrics
    feature_columns = [
        'sessionDuration',  # Time spent on website
        'pageviews',  # Number of pages viewed
        'bounces',  # Single-page sessions
        'sessions',  # Number of sessions
        'goal4Completions',  # Goal completions
        'newUser',  # New vs returning user
        'visited_demo_page',  # Visited product demo page
        'visited_checkout_page',  # Visited checkout page
        'visited_water_purifier_page',  # Visited water purifier page
        'fired_phone_clicks_evt',  # Clicked phone number
        'fired_help_me_buy_evt',  # Clicked help me buy
        'DemoReqPg_CallClicks_evt_count',  # Demo request call clicks
        'help_me_buy_evt_count',  # Help me buy event count
        'phone_clicks_evt_count',  # Phone click event count
        'paid'  # Paid traffic indicator
    ]

    # Ensure all selected features exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"\nAvailable numerical features for modeling: {len(available_features)}")

    # Handle categorical variables by creating dummy variables
    # This converts categorical data into numerical format for ML models
    categorical_cols = ['device', 'sourceMedium', 'country', 'region']
    print(f"\nProcessing categorical variables...")

    for col in categorical_cols:
        if col in df.columns:
            print(f"  - Creating dummies for {col} ({df[col].nunique()} unique values)")
            # Create dummy variables (one-hot encoding)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            available_features.extend(dummies.columns.tolist())

    # Prepare final feature set
    X = df[available_features].copy()
    y = df['converted_binary'].copy()

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")

    return X, y, df


# STEP 2: MODEL TRAINING FUNCTIONS
# ===============================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Logistic Regression model

    Logistic Regression is a linear model good for:
    - Interpretable coefficients
    - Fast training and prediction
    - Baseline model performance

    Uses StandardScaler for feature normalization and class_weight='balanced'
    to handle class imbalance.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    # Scale features for logistic regression (important for convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build logistic regression model with class balancing
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handles class imbalance
    )
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1_score': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }

    print("Logistic Regression Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance analysis (coefficients)
    feature_importance_lr = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print("\nTop 10 Most Important Features (Logistic Regression):")
    for idx, row in feature_importance_lr.head(10).iterrows():
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"  {row['feature']}: {row['coefficient']:.4f} ({direction} conversion probability)")

    # Confusion matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_lr[0, 0]:,}")
    print(f"  False Positives: {cm_lr[0, 1]:,}")
    print(f"  False Negatives: {cm_lr[1, 0]:,}")
    print(f"  True Positives: {cm_lr[1, 1]:,}")

    return results, lr_model, scaler


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Decision Tree model

    Decision Trees are good for:
    - Non-linear relationships
    - Feature interactions
    - Interpretable rules
    - No need for feature scaling

    Uses hyperparameters to prevent overfitting.
    """
    print("\n" + "=" * 60)
    print("DECISION TREE CLASSIFIER")
    print("=" * 60)

    # Build decision tree model with regularization parameters
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=100,  # Minimum samples to split a node
        class_weight='balanced'  # Handle class imbalance
    )
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Decision Tree',
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt),
        'recall': recall_score(y_test, y_pred_dt),
        'f1_score': f1_score(y_test, y_pred_dt),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
    }

    print("Decision Tree Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance (based on information gain)
    feature_importance_dt = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Decision Tree):")
    for idx, row in feature_importance_dt.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_dt[0, 0]:,}")
    print(f"  False Positives: {cm_dt[0, 1]:,}")
    print(f"  False Negatives: {cm_dt[1, 0]:,}")
    print(f"  True Positives: {cm_dt[1, 1]:,}")

    return results, dt_model


def train_random_forest_smote(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest model with SMOTE sampling

    Random Forest combines multiple decision trees and is good for:
    - Handling overfitting better than single trees
    - Feature importance ranking
    - Robust performance

    SMOTE (Synthetic Minority Oversampling Technique) creates synthetic
    examples of the minority class to balance the dataset.
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL WITH SMOTE SAMPLING")
    print("=" * 60)

    # Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  SMOTE balanced training set: {X_train_smote.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

    # Build Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Maximum depth of trees
        min_samples_split=20,  # Minimum samples to split a node
        min_samples_leaf=10,  # Minimum samples in a leaf
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    rf_model.fit(X_train_smote, y_train_smote)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Random Forest (SMOTE)',
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }

    print("\nRandom Forest Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Random Forest):")
    for idx, row in feature_importance_rf.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_rf[0, 0]:,}")
    print(f"  False Positives: {cm_rf[0, 1]:,}")
    print(f"  False Negatives: {cm_rf[1, 0]:,}")
    print(f"  True Positives: {cm_rf[1, 1]:,}")

    return results, rf_model


def train_xgboost_oversampling(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost model with Random Oversampling

    XGBoost is a gradient boosting framework that is excellent for:
    - High performance on structured data
    - Built-in regularization
    - Feature importance
    - Handling missing values

    Random Oversampling duplicates minority class samples to balance the dataset.
    """
    print("\n" + "=" * 60)
    print("XGBOOST MODEL WITH RANDOM OVERSAMPLING")
    print("=" * 60)

    # Apply random oversampling for minority class
    print("Applying Random Oversampling to balance classes...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  Up-sampled training set: {X_train_ros.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Up-sampled class distribution: {pd.Series(y_train_ros).value_counts().to_dict()}")

    # Build XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,  # Number of boosting rounds
        max_depth=6,  # Maximum depth of trees
        learning_rate=0.1,  # Step size shrinkage
        subsample=0.8,  # Fraction of samples used for training each tree
        colsample_bytree=0.8,  # Fraction of features used for training each tree
        random_state=42,
        eval_metric='logloss'  # Evaluation metric
    )

    xgb_model.fit(X_train_ros, y_train_ros)

    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'XGBoost (Random Oversampling)',
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1_score': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb)
    }

    print("\nXGBoost Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_xgb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (XGBoost):")
    for idx, row in feature_importance_xgb.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_xgb[0, 0]:,}")
    print(f"  False Positives: {cm_xgb[0, 1]:,}")
    print(f"  False Negatives: {cm_xgb[1, 0]:,}")
    print(f"  True Positives: {cm_xgb[1, 1]:,}")

    return results, xgb_model


def train_gradient_boosting_smote(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Gradient Boosting model with SMOTE sampling

    Gradient Boosting builds models sequentially, where each model
    corrects the errors of the previous ones. Good for:
    - High predictive accuracy
    - Handling complex patterns
    - Feature importance
    """
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING MODEL WITH SMOTE SAMPLING")
    print("=" * 60)

    # Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  Original training set: {X_train.shape[0]:,} samples")
    print(f"  SMOTE balanced training set: {X_train_smote.shape[0]:,} samples")
    print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"  SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

    # Build Gradient Boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=100,  # Number of boosting stages
        max_depth=6,  # Maximum depth of trees
        learning_rate=0.1,  # Learning rate shrinks contribution of each tree
        subsample=0.8,  # Fraction of samples used for fitting trees
        random_state=42
    )

    gb_model.fit(X_train_smote, y_train_smote)

    # Make predictions
    y_pred_gb = gb_model.predict(X_test)
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

    # Evaluate model performance
    results = {
        'model': 'Gradient Boosting (SMOTE)',
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1_score': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }

    print("\nGradient Boosting Results:")
    for metric, value in results.items():
        if metric != 'model':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Feature importance
    feature_importance_gb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Gradient Boosting):")
    for idx, row in feature_importance_gb.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Confusion matrix
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm_gb[0, 0]:,}")
    print(f"  False Positives: {cm_gb[0, 1]:,}")
    print(f"  False Negatives: {cm_gb[1, 0]:,}")
    print(f"  True Positives: {cm_gb[1, 1]:,}")

    return results, gb_model


# STEP 3: VISUALIZATION FUNCTIONS
# ==============================

def create_model_comparison_plots(all_results):
    """
    Create comprehensive visualizations comparing all models

    This function creates:
    1. Model performance comparison bar chart
    2. ROC curves for all models
    3. Feature importance comparison
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Eureka Forbes Conversion Prediction - Model Comparison', fontsize=16, fontweight='bold')

    # 1. Model Performance Comparison
    metrics_df = pd.DataFrame(all_results)
    metrics_df = metrics_df.set_index('model')

    # Plot performance metrics
    ax1 = axes[0, 0]
    metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].plot(kind='bar', ax=ax1)
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. F1-Score vs ROC-AUC Scatter Plot
    ax2 = axes[0, 1]
    scatter = ax2.scatter(metrics_df['f1_score'], metrics_df['roc_auc'],
                          s=100, alpha=0.7, c=range(len(metrics_df)), cmap='viridis')

    # Add model labels to points
    for i, model in enumerate(metrics_df.index):
        ax2.annotate(model, (metrics_df.iloc[i]['f1_score'], metrics_df.iloc[i]['roc_auc']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('F1-Score')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('F1-Score vs ROC-AUC', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    scatter2 = ax3.scatter(metrics_df['recall'], metrics_df['precision'],
                           s=100, alpha=0.7, c=range(len(metrics_df)), cmap='plasma')

    # Add model labels to points
    for i, model in enumerate(metrics_df.index):
        ax3.annotate(model, (metrics_df.iloc[i]['recall'], metrics_df.iloc[i]['precision']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Model Ranking by Different Metrics
    ax4 = axes[1, 1]

    # Create ranking for each metric
    ranking_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        ranked = metrics_df[metric].rank(ascending=False)
        for model, rank in ranked.items():
            ranking_data.append({'Model': model, 'Metric': metric, 'Rank': rank})

    ranking_df = pd.DataFrame(ranking_data)
    ranking_pivot = ranking_df.pivot(index='Model', columns='Metric', values='Rank')

    # Create heatmap
    sns.heatmap(ranking_pivot, annot=True, cmap='RdYlGn_r', ax=ax4,
                cbar_kws={'label': 'Rank (1=Best)'})
    ax4.set_title('Model Ranking by Metric', fontweight='bold')
    ax4.set_ylabel('')

    plt.tight_layout()
    plt.savefig('eureka_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics_df


def plot_feature_importance_comparison(feature_importances_dict):
    """
    Compare feature importance across different models
    """
    num_models = len(feature_importances_dict)

    # Calculate appropriate grid dimensions
    if num_models <= 2:
        rows, cols = 1, num_models
    elif num_models <= 4:
        rows, cols = 2, 2
    elif num_models <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    fig.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')

    # Handle single subplot case
    if num_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()

    model_names = list(feature_importances_dict.keys())

    for i, (model_name, importance_df) in enumerate(feature_importances_dict.items()):
        ax = axes[i]

        # Get top 10 features
        top_features = importance_df.head(10)

        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features.iloc[:, 1],
                       color=plt.cm.Set3(i))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.iloc[:, 0])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top 10 Features', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)

    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('eureka_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_data):
    """
    Plot ROC curves for all models on the same plot
    """
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (model_name, y_test, y_pred_proba) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('eureka_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_data_exploration_plots(df):
    """
    Create exploratory data analysis plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Eureka Forbes Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # 1. Conversion rate by device
    ax1 = axes[0, 0]
    conversion_by_device = df.groupby('device')['converted_binary'].agg(['count', 'sum', 'mean'])
    conversion_by_device['conversion_rate'] = conversion_by_device['mean'] * 100

    bars = ax1.bar(conversion_by_device.index, conversion_by_device['conversion_rate'])
    ax1.set_title('Conversion Rate by Device Type')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_xlabel('Device Type')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 2. Session duration distribution
    ax2 = axes[0, 1]
    ax2.hist(df['sessionDuration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Session Duration Distribution')
    ax2.set_xlabel('Session Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['sessionDuration'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["sessionDuration"].mean():.0f}s')
    ax2.legend()

    # 3. Pageviews vs Conversion
    ax3 = axes[0, 2]
    converted = df[df['converted_binary'] == 1]['pageviews']
    not_converted = df[df['converted_binary'] == 0]['pageviews']

    ax3.hist([not_converted, converted], bins=30, alpha=0.7,
             label=['Not Converted', 'Converted'], color=['lightcoral', 'lightgreen'])
    ax3.set_title('Pageviews Distribution by Conversion')
    ax3.set_xlabel('Number of Pageviews')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_yscale('log')

    # 4. Traffic source analysis
    ax4 = axes[1, 0]
    source_conversion = df.groupby('sourceMedium')['converted_binary'].agg(['count', 'mean'])
    source_conversion = source_conversion[source_conversion['count'] >= 100]  # Filter for significant sources
    source_conversion = source_conversion.sort_values('mean', ascending=True)

    bars = ax4.barh(range(len(source_conversion)), source_conversion['mean'] * 100)
    ax4.set_yticks(range(len(source_conversion)))
    ax4.set_yticklabels([src[:20] + '...' if len(src) > 20 else src for src in source_conversion.index])
    ax4.set_xlabel('Conversion Rate (%)')
    ax4.set_title('Conversion Rate by Traffic Source')

    # 5. New vs Returning Users
    ax5 = axes[1, 1]
    user_type_conv = df.groupby('newUser')['converted_binary'].agg(['count', 'mean'])
    user_labels = ['Returning User', 'New User']

    bars = ax5.bar(user_labels, user_type_conv['mean'] * 100, color=['orange', 'green'])
    ax5.set_title('Conversion Rate: New vs Returning Users')
    ax5.set_ylabel('Conversion Rate (%)')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 6. Correlation heatmap of key features
    ax6 = axes[1, 2]
    key_features = ['sessionDuration', 'pageviews', 'sessions', 'bounces',
                    'goal4Completions', 'converted_binary']
    correlation_matrix = df[key_features].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('eureka_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()


# STEP 4: MAIN EXECUTION FUNCTION
# ===============================

def main():
    """
    Main function that executes the complete machine learning pipeline

    This function:
    1. Loads and preprocesses the data
    2. Splits data into train/test sets
    3. Trains all models with appropriate sampling techniques
    4. Compares model performance
    5. Creates comprehensive visualizations
    6. Provides recommendations
    """
    print("=" * 80)
    print("EUREKA FORBES CONVERSION PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    file_path = 'eureka_data_final_2019-01-01_2019-03-01.csv'  # Update path as needed
    X, y, df = load_and_preprocess_data(file_path)

    # Step 2: Split data into training and testing sets
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA INTO TRAIN/TEST SETS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]:,} samples")
    print(f"Testing set size: {X_test.shape[0]:,} samples")
    print(f"Training set conversion rate: {y_train.mean() * 100:.3f}%")
    print(f"Testing set conversion rate: {y_test.mean() * 100:.3f}%")

    # Step 3: Train all models
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING ALL MODELS")
    print("=" * 60)

    all_results = []
    trained_models = {}
    feature_importances = {}
    roc_data = []

    # 1. Logistic Regression
    try:
        lr_results, lr_model, lr_scaler = train_logistic_regression(X_train, X_test, y_train, y_test)
        all_results.append(lr_results)
        trained_models['Logistic Regression'] = (lr_model, lr_scaler)

        # Get feature importance for logistic regression
        feature_importance_lr = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(lr_model.coef_[0])
        }).sort_values('importance', ascending=False)
        feature_importances['Logistic Regression'] = feature_importance_lr

        # ROC data
        y_pred_proba_lr = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
        roc_data.append(('Logistic Regression', y_test, y_pred_proba_lr))

    except Exception as e:
        print(f"Error training Logistic Regression: {e}")

    # 2. Decision Tree
    try:
        dt_results, dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
        all_results.append(dt_results)
        trained_models['Decision Tree'] = dt_model

        # Get feature importance for decision tree
        feature_importance_dt = pd.DataFrame({
            'feature': X_train.columns,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Decision Tree'] = feature_importance_dt

        # ROC data
        y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Decision Tree', y_test, y_pred_proba_dt))

    except Exception as e:
        print(f"Error training Decision Tree: {e}")

    # 3. Random Forest with SMOTE
    try:
        rf_results, rf_model = train_random_forest_smote(X_train, X_test, y_train, y_test)
        all_results.append(rf_results)
        trained_models['Random Forest (SMOTE)'] = rf_model

        # Get feature importance for random forest
        feature_importance_rf = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Random Forest (SMOTE)'] = feature_importance_rf

        # ROC data
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Random Forest (SMOTE)', y_test, y_pred_proba_rf))

    except Exception as e:
        print(f"Error training Random Forest: {e}")

    # 4. XGBoost with Random Oversampling
    try:
        xgb_results, xgb_model = train_xgboost_oversampling(X_train, X_test, y_train, y_test)
        all_results.append(xgb_results)
        trained_models['XGBoost (Random Oversampling)'] = xgb_model

        # Get feature importance for XGBoost
        feature_importance_xgb = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['XGBoost (Random Oversampling)'] = feature_importance_xgb

        # ROC data
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        roc_data.append(('XGBoost (Random Oversampling)', y_test, y_pred_proba_xgb))

    except Exception as e:
        print(f"Error training XGBoost: {e}")

    # 5. Gradient Boosting with SMOTE
    try:
        gb_results, gb_model = train_gradient_boosting_smote(X_train, X_test, y_train, y_test)
        all_results.append(gb_results)
        trained_models['Gradient Boosting (SMOTE)'] = gb_model

        # Get feature importance for gradient boosting
        feature_importance_gb = pd.DataFrame({
            'feature': X_train.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importances['Gradient Boosting (SMOTE)'] = feature_importance_gb

        # ROC data
        y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
        roc_data.append(('Gradient Boosting (SMOTE)', y_test, y_pred_proba_gb))

    except Exception as e:
        print(f"Error training Gradient Boosting: {e}")

    # Step 4: Model Comparison and Visualization
    print("\n" + "=" * 60)
    print("STEP 4: MODEL COMPARISON AND VISUALIZATION")
    print("=" * 60)

    if all_results:
        # Create comparison table
        comparison_df = create_model_comparison_plots(all_results)

        # Display final results table
        print("\nFINAL MODEL COMPARISON:")
        print("=" * 80)
        print(comparison_df.round(4))

        # Find best model for each metric
        print("\nBEST PERFORMING MODELS BY METRIC:")
        print("=" * 50)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")

        # Create feature importance comparison
        if feature_importances:
            plot_feature_importance_comparison(feature_importances)

        # Create ROC curves
        if roc_data:
            plot_roc_curves(roc_data)

        # Create data exploration plots
        create_data_exploration_plots(df)

        # Step 5: Recommendations
        print("\n" + "=" * 60)
        print("STEP 5: RECOMMENDATIONS AND INSIGHTS")
        print("=" * 60)

        # Overall best model recommendation
        best_f1_model = comparison_df['f1_score'].idxmax()
        best_auc_model = comparison_df['roc_auc'].idxmax()

        print("BUSINESS RECOMMENDATIONS:")
        print("=" * 30)
        print(f"1. BEST OVERALL MODEL: {best_f1_model}")
        print(f"   - F1-Score: {comparison_df.loc[best_f1_model, 'f1_score']:.4f}")
        print(f"   - ROC-AUC: {comparison_df.loc[best_f1_model, 'roc_auc']:.4f}")
        print(f"   - This model provides the best balance of precision and recall")

        print(f"\n2. BEST FOR RANKING/SCORING: {best_auc_model}")
        print(f"   - ROC-AUC: {comparison_df.loc[best_auc_model, 'roc_auc']:.4f}")
        print(f"   - Use this model for probability scoring and ranking customers")

        # Feature insights
        if feature_importances:
            print("\n3. KEY CONVERSION DRIVERS:")
            # Get most common top features across models
            all_top_features = []
            for model_name, importance_df in feature_importances.items():
                all_top_features.extend(importance_df.head(5)['feature'].tolist())

            from collections import Counter
            feature_counts = Counter(all_top_features)

            print("   Most important features across all models:")
            for feature, count in feature_counts.most_common(10):
                print(f"   - {feature} (mentioned in {count} models)")

        print("\n4. BUSINESS ACTIONS:")
        print("   - Focus on improving session duration and page engagement")
        print("   - Optimize demo page and checkout page experience")
        print("   - Implement targeted campaigns for high-value traffic sources")
        print("   - Use model predictions to prioritize sales follow-ups")

        # Save results to CSV
        comparison_df.to_csv('eureka_model_results.csv')
        print("\n5. OUTPUTS SAVED:")
        print("   - eureka_model_results.csv: Model performance comparison")
        print("   - eureka_model_comparison.png: Performance visualization")
        print("   - eureka_feature_importance_comparison.png: Feature importance")
        print("   - eureka_roc_curves.png: ROC curve comparison")
        print("   - eureka_data_exploration.png: Data exploration plots")

        return trained_models, comparison_df, feature_importances

    else:
        print("No models were successfully trained. Please check the data and try again.")
        return None, None, None


# EXECUTION
# =========
if __name__ == "__main__":
    # Run the complete pipeline
    models, results, importances = main()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nTo use this script:")
    print("1. Ensure all required packages are installed")
    print("2. Update the file_path variable to point to your CSV file")
    print("3. Run: python eureka_complete_script.py")










