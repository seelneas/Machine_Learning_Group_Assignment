import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle  
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('ML Regression Problem/AMZN_stock_data.csv', parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)

# Display basic information about the dataset
print("*******************************")
print("First 5 rows of the dataset\n")
print("*******************************\n")
print(df.head())
print("\n*******************************")
print("\nDataset information\n")
print("*******************************\n")
print(df.info())
print("\n*******************************")
print("\nDescriptive statistics\n")
print("*******************************\n")
print(df.describe())
print("\n")

# Create features using lagged closing prices
lookback = 3  # Using 3 previous days to predict next day
for i in range(1, lookback+1):
    df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
df.dropna(inplace=True)

X = df[['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']]
y = df['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1)
}

results = {}

# Train and evaluate models, saving them to .pkl files
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train_scaled.ravel())
    
    # Save the model to a .pkl file
    file_path = f'ML Regression Problem/{name.replace(" ", "_")}.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    
    # Load the model from the .pkl file (to demonstrate usage)
    with open(f'ML Regression Problem/{name.replace(" ", "_")}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Prediction using the loaded model
    y_pred_scaled = loaded_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R² Score': r2,
        'Predictions': y_pred
    }

# Create comparison table
print("*******************************\n")
print("Regression Models Comparison Table")
print("\n*******************************\n")
comparison_df = pd.DataFrame.from_dict(results, orient='index')
metrics_df = comparison_df[['MAE', 'MSE', 'RMSE', 'R² Score']]
print(metrics_df)

# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(y_test.index, y_test, label='Actual Prices', linewidth=2)

for model_name, result in results.items():
    plt.plot(y_test.index, result['Predictions'], '--', label=f'{model_name} Predictions')

plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
