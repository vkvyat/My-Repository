import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

#Load and merge Zillow datasets
def load_and_merge_data():

    print("Loading datasets...")

    # Load main datasets
    train_2016 = pd.read_csv('train_2016_v2.csv')
    train_2017 = pd.read_csv('train_2017.csv')
    properties_2016 = pd.read_csv('properties_2016.csv')
    properties_2017 = pd.read_csv('properties_2017.csv')

    # Combine training data
    train_combined = pd.concat([train_2016, train_2017], ignore_index=True)

    # Combine property data
    properties_combined = pd.concat([properties_2016, properties_2017], ignore_index=True)
    properties_combined = properties_combined.drop_duplicates(subset=['parcelid'], keep='last')

    # Merge training data with properties
    master_data = train_combined.merge(properties_combined, on='parcelid', how='inner')

    print(f"Master dataset shape: {master_data.shape}")
    return master_data


def clean_and_engineer_features(df):
    """Clean data and engineer features"""
    print("Cleaning data and engineering features...")

    # Create copy for processing
    df_clean = df.copy()

    # Handle missing values for key features - only use columns that exist
    available_cols = df_clean.columns.tolist()
    numeric_cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
                    'taxvaluedollarcnt', 'yearbuilt', 'latitude', 'longitude']

    # Add lotsize only if it exists
    if 'lotsize' in available_cols:
        numeric_cols.append('lotsize')
    if 'lotsizesquarefeet' in available_cols:
        numeric_cols.append('lotsizesquarefeet')

    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill categorical missing values
    categorical_cols = ['airconditioningtypeid', 'heatingorsystemtypeid', 'regionidzip']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)

    # Feature engineering
    current_year = 2017
    df_clean['property_age'] = current_year - df_clean['yearbuilt']
    df_clean['property_age'] = df_clean['property_age'].clip(0, 150)

    # Tax ratio features
    df_clean['tax_per_sqft'] = df_clean['taxvaluedollarcnt'] / df_clean['calculatedfinishedsquarefeet']
    df_clean['tax_per_sqft'] = df_clean['tax_per_sqft'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Room ratios
    df_clean['bath_bed_ratio'] = df_clean['bathroomcnt'] / df_clean['bedroomcnt']
    df_clean['bath_bed_ratio'] = df_clean['bath_bed_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)

    # Size categories
    df_clean['size_category'] = pd.cut(df_clean['calculatedfinishedsquarefeet'],
                                       bins=[0, 1000, 2000, 3000, np.inf],
                                       labels=['Small', 'Medium', 'Large', 'XLarge'])

    # AC type categories
    ac_mapping = {1: 'Central', 5: 'Other', 13: 'None'}
    df_clean['ac_type'] = df_clean['airconditioningtypeid'].map(ac_mapping).fillna('Unknown')

    # Remove extreme outliers
    df_clean = df_clean[
        (df_clean['logerror'].abs() <= 0.5) &
        (df_clean['bedroomcnt'] <= 10) &
        (df_clean['bathroomcnt'] <= 8) &
        (df_clean['calculatedfinishedsquarefeet'] <= 10000) &
        (df_clean['calculatedfinishedsquarefeet'] >= 500)
        ]

    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

# Building Random Forest model and generating predictions
def build_prediction_model(df):

    print("Building prediction model...")

    # Select features for modeling - only use columns that exist
    base_features = [
        'bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
        'taxvaluedollarcnt', 'yearbuilt', 'latitude', 'longitude',
        'property_age', 'tax_per_sqft', 'bath_bed_ratio', 'airconditioningtypeid'
    ]

    # Add lot size if available
    if 'lotsize' in df.columns:
        base_features.append('lotsize')
    elif 'lotsizesquarefeet' in df.columns:
        base_features.append('lotsizesquarefeet')

    # Filter to only existing columns
    feature_cols = [col for col in base_features if col in df.columns]

    # Prepare data
    model_data = df[feature_cols + ['logerror']].dropna()
    X = model_data[feature_cols]
    y = model_data['logerror']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Generate predictions
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance - MAE: {mae:.4f}, R²: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'actual_logerror': y_test,
        'predicted_logerror': y_pred,
        'prediction_error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })

    return rf_model, feature_importance, predictions_df, mae, r2

# Creating master dataset for Tableau import
def create_master_dataset(df, feature_importance, predictions_df, mae, r2):

    print("Creating master dataset for Tableau...")

    # Select key columns for master dataset - only use existing columns
    base_cols = [
        'parcelid', 'logerror', 'transactiondate',
        'bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet',
        'taxvaluedollarcnt', 'yearbuilt',
        'latitude', 'longitude', 'regionidzip',
        'airconditioningtypeid', 'heatingorsystemtypeid',
        'property_age', 'tax_per_sqft', 'bath_bed_ratio',
        'size_category', 'ac_type'
    ]

    # Add lot size if available
    if 'lotsize' in df.columns:
        base_cols.append('lotsize')
    elif 'lotsizesquarefeet' in df.columns:
        base_cols.append('lotsizesquarefeet')

    # Filter to only existing columns
    master_cols = [col for col in base_cols if col in df.columns]

    # Create master dataset
    master_dataset = df[master_cols].copy()

    # Add model performance metrics as columns
    master_dataset['model_mae'] = mae
    master_dataset['model_r2'] = r2

    # Add error categories
    master_dataset['error_category'] = pd.cut(
        master_dataset['logerror'].abs(),
        bins=[0, 0.05, 0.1, 0.2, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # Add geographic regions based on latitude/longitude
    master_dataset['lat_region'] = pd.cut(master_dataset['latitude'], bins=5,
                                          labels=['South', 'South-Mid', 'Central', 'North-Mid', 'North'])
    master_dataset['lon_region'] = pd.cut(master_dataset['longitude'], bins=5,
                                          labels=['West', 'West-Mid', 'Central', 'East-Mid', 'East'])

    # Remove any remaining missing values
    master_dataset = master_dataset.dropna()

    print(f"Final master dataset shape: {master_dataset.shape}")
    return master_dataset

# Saving output files to the directory
def save_all_outputs(master_dataset, feature_importance, predictions_df):

    print("Saving output files...")

    # Save master dataset
    master_dataset.to_csv('master_dataset_tableau.csv', index=False)
    master_dataset.to_excel('master_dataset_tableau.xlsx', index=False, engine='openpyxl')

    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    feature_importance.to_excel('feature_importance.xlsx', index=False, engine='openpyxl')

    # Save model predictions
    predictions_df.to_csv('model_predictions.csv', index=False)
    predictions_df.to_excel('model_predictions.xlsx', index=False, engine='openpyxl')

    print("All files saved successfully!")

    return {
        'master_dataset': 'master_dataset_tableau.xlsx',
        'feature_importance': 'feature_importance.xlsx',
        'model_predictions': 'model_predictions.xlsx'
    }

# Establish main execution function
def main():

    print("=" * 60)
    print("ZILLOW HOME VALUE PREDICTION - DATA PROCESSING PIPELINE")
    print("=" * 60)

    try:
        # Step 1: Load and merge data
        master_data = load_and_merge_data()

        # Step 2: Clean and engineer features
        clean_data = clean_and_engineer_features(master_data)

        # Step 3: Build prediction model
        model, feature_importance, predictions_df, mae, r2 = build_prediction_model(clean_data)

        # Step 4: Create master dataset
        master_dataset = create_master_dataset(clean_data, feature_importance, predictions_df, mae, r2)

        # Step 5: Save all outputs
        output_files = save_all_outputs(master_dataset, feature_importance, predictions_df)

        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Master Dataset Shape: {master_dataset.shape}")
        print(f"Model MAE: {mae:.4f}")
        print(f"Model R²: {r2:.4f}")
        print("\nOutput Files Created:")
        for key, filename in output_files.items():
            print(f"  - {key}: {filename}")

        print("\nMaster Dataset Preview:")
        print(master_dataset.head())

        return master_dataset, feature_importance, predictions_df

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    master_dataset, feature_importance, predictions_df = main()

