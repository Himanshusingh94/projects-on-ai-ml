import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Load Dataset

df = pd.read_csv('vgchartz-2024.csv')

# Drop rows where target (total_sales) is missing
df = df.dropna(subset=['total_sales']).reset_index(drop=True)

# Define target and features
target_column = 'total_sales'
feature_columns = [col for col in df.columns if col != target_column]

X = df[feature_columns]
y = df[target_column]

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# Preprocessing Pipelines

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# Train-Test Split Variation
split_ratios = [0.1, 0.2, 0.3, 0.4]
all_results = []

graph_folder = "graphs"
os.makedirs(graph_folder, exist_ok=True)
sns.set(style="whitegrid")

for test_size in split_ratios:
    print(f"\nEvaluating for test size = {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- Simple Linear Regression (per numeric feature) ---
    for feature in numeric_features:
        pipe = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', numeric_transformer, [feature])
            ])),
            ('regressor', LinearRegression())
        ])
        pipe.fit(X_train[[feature]], y_train)
        y_pred = pipe.predict(X_test[[feature]])

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        all_results.append({
            'Model': f'SimpleLinear_{feature}',
            'Test Size': test_size,
            'Test MSE': mse,
            'Test MAE': mae,
            'Test R2': r2
        })

    # --- Multiple Linear Regression (all features) ---
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    all_results.append({
        'Model': 'MultipleLinearRegression',
        'Test Size': test_size,
        'Test MSE': mse,
        'Test MAE': mae,
        'Test R2': r2
    })

    # --- Regression Plot (Actual vs Predicted) ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Total Sales')
    plt.ylabel('Predicted Total Sales')
    plt.title(f'Regression Plot (Actual vs Predicted) - Test Size {test_size}')
    plt.tight_layout()
    plot_path = f"{graph_folder}/Regression_Plot_TestSize_{test_size}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved regression plot: {plot_path}")

# Save Results
results_df = pd.DataFrame(all_results)
results_df.to_csv('all_models_train_test_split_results.csv', index=False)
print("Saved all models train-test split results to CSV.")

# Metric Plots
metrics = ['Test MSE', 'Test MAE', 'Test R2']
for metric in metrics:
    plt.figure(figsize=(10, 6))

    if metric == 'Test R2':
        sns.lineplot(data=results_df, x='Test Size', y=metric, hue='Model', marker="o")
        plt.ylim(0, 1.05)
    else:
        sns.barplot(data=results_df, x='Test Size', y=metric, hue='Model')

    plt.title(f'{metric} Across Different Train-Test Splits By Model')
    plt.xlabel('Test Size')
    plt.ylabel(metric)
    plt.tight_layout()

    graph_path = f"{graph_folder}/{metric.replace(' ', '_')}.png"
    plt.savefig(graph_path)
    plt.close()
    print(f"Saved plot: {graph_path}")

# Zip Graphs

shutil.make_archive('graphs', 'zip', graph_folder)
print("Graphs zipped into graphs.zip.")
