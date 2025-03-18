# Basic imports
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# MLP imports
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Save/load models
from joblib import dump, load
import matplotlib.pyplot as plt

def load_and_prepare_data(scaled_data_path="2_scaled_data.csv"):
    """Load scaled data and prepare X, y datasets"""
    df = pd.read_csv(scaled_data_path)
    df = df.dropna()
    
    # Get the X and y from already scaled data
    X = df[["cost", "productivity", "area"]]
    y = df["value"]
    
    return X, y, df

def create_mlp_model(random_state=42):
    """Create MLPRegressor with optimized parameters"""
    return MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=10000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state
    )

def print_metrics(y_true, y_pred, dataset_name=""):
    """Calculate and print model metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def plot_results(y_train, y_pred_train, y_test, y_pred_test):
    """Plot prediction results and residuals and save them"""
    # Prediction plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Training: Actual vs. Predicted Values')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Test: Actual vs. Predicted Values')

    plt.tight_layout()
    plt.savefig('mlp_results/prediction_plots.png')

    # Residual plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    residuals_train = y_train - y_pred_train
    plt.scatter(y_pred_train, residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis - Training')

    plt.subplot(1, 2, 2)
    residuals_test = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals_test, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis - Test')

    plt.tight_layout()
    plt.savefig('mlp_results/residual_plots.png')

def plot_residual_analysis(y_train, y_pred_train, y_test, y_pred_test):
    """Additional residual analysis plots"""
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    # Standardize residuals
    std_residuals_train = (residuals_train - residuals_train.mean()) / residuals_train.std()
    std_residuals_test = (residuals_test - residuals_test.mean()) / residuals_test.std()
    
    # Histogram of residuals
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals_train, bins=30, alpha=0.7, color='blue', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Training Residuals Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals_test, bins=30, alpha=0.7, color='blue', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Test Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig('mlp_results/residuals_distribution.png')
    
    # Q-Q plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    stats.probplot(std_residuals_train, dist="norm", plot=plt)
    plt.title('Training Q-Q Plot')
    
    plt.subplot(1, 2, 2)
    stats.probplot(std_residuals_test, dist="norm", plot=plt)
    plt.title('Test Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig('mlp_results/qq_plots.png')
    
    # Standardized residuals vs Predicted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred_train, std_residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=2, color='r', linestyle=':')
    plt.axhline(y=-2, color='r', linestyle=':')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Training: Standardized Residuals vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_test, std_residuals_test, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=2, color='r', linestyle=':')
    plt.axhline(y=-2, color='r', linestyle=':')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Test: Standardized Residuals vs Predicted')
    
    plt.tight_layout()
    plt.savefig('mlp_results/standardized_residuals.png')

def plot_feature_importance(model, feature_names):
    """Plot feature importance based on neural network weights"""
    # Get the absolute weights from first layer
    weights = np.abs(model.coefs_[0])
    importance = np.mean(weights, axis=1)
    
    # Normalize importance
    importance = importance / np.sum(importance)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mlp_results/feature_importance.png')

def print_analysis_results(y_train, y_pred_train, y_test, y_pred_test, X_train):
    """Print statistical analysis of the model results"""
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    # Standardize residuals
    std_residuals_train = (residuals_train - residuals_train.mean()) / residuals_train.std()
    std_residuals_test = (residuals_test - residuals_test.mean()) / residuals_test.std()
    
    print("\n=== Residuals Analysis ===")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Training Residuals - Mean: {residuals_train.mean():.4f}, Std: {residuals_train.std():.4f}")
    print(f"Test Residuals - Mean: {residuals_test.mean():.4f}, Std: {residuals_test.std():.4f}")
    
    # Normality tests
    print("\nNormality Tests (Shapiro-Wilk):")
    _, p_value_train = stats.shapiro(residuals_train)
    _, p_value_test = stats.shapiro(residuals_test)
    print(f"Training Residuals - p-value: {p_value_train:.4f}")
    print(f"Test Residuals - p-value: {p_value_test:.4f}")
    print("Interpretation: p-value > 0.05 suggests normal distribution")
    
    # Outliers analysis
    train_outliers = np.sum(np.abs(std_residuals_train) > 2)
    test_outliers = np.sum(np.abs(std_residuals_test) > 2)
    print("\nOutliers Analysis (|standardized residuals| > 2):")
    print(f"Training set: {train_outliers} outliers ({train_outliers/len(y_train)*100:.1f}% of data)")
    print(f"Test set: {test_outliers} outliers ({test_outliers/len(y_test)*100:.1f}% of data)")
    
    # Homoscedasticity (Breusch-Pagan test)
    X_train_const = sm.add_constant(X_train)  # Add constant column for the test
    _, p_value_bp_train, _, _ = het_breuschpagan(residuals_train, X_train_const)
    print("\nHomoscedasticity Test (Breusch-Pagan):")
    print(f"Training set - p-value: {p_value_bp_train:.4f}")
    print("Interpretation: p-value > 0.05 suggests homoscedasticity (constant variance)")

def main():
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("mlp_results").mkdir(exist_ok=True)
    
    # Load scaled data
    X, y, df = load_and_prepare_data()
    
    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    regr = create_mlp_model()
    regr.fit(X_train, y_train)
    
    # Evaluate model
    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    print(f"R² Score (Train): {train_score:.4f}")
    print(f"R² Score (Test): {test_score:.4f}")
    
    # Cross-validation
    scores = cross_val_score(regr, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Average CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Make predictions
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)
    
    # Calculate and print metrics
    print_metrics(y_train, y_pred_train, "Treino")
    print_metrics(y_test, y_pred_test, "Teste")
    
    # Plot results and print analysis
    plot_results(y_train, y_pred_train, y_test, y_pred_test)
    plot_residual_analysis(y_train, y_pred_train, y_test, y_pred_test)
    plot_feature_importance(regr, X.columns)
    print_analysis_results(y_train, y_pred_train, y_test, y_pred_test, X_train)
    
    # Save predictions to dataframe
    df["predicted_value"] = regr.predict(X)
    df.to_csv("3_data_with_predictions.csv", index=False)
    
    # Save the model
    dump(regr, 'models/mlp.joblib')

if __name__ == "__main__":
    main()