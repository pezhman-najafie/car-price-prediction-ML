from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('cardata.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Car_Name'] = label_encoder.fit_transform(df['Car_Name'])
df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = label_encoder.fit_transform(df['Seller_Type'])
df['Transmission'] = label_encoder.fit_transform(df['Transmission'])

# Split data into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# DataFrame 1: Min-Max Scaling
min_max_scaler = MinMaxScaler()
train_data_minmax = min_max_scaler.fit_transform(train_data.drop('Selling_Price', axis=1))
test_data_minmax = min_max_scaler.transform(test_data.drop('Selling_Price', axis=1))

# Function to train model and display evaluation metrics
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)

    # Plot predicted vs actual
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Selling Price")
    plt.show()

# Train and evaluate model with Min-Max Scaling data
print("Results for Min-Max Scaling:")
train_and_evaluate_model(train_data_minmax, test_data_minmax, train_data['Selling_Price'], test_data['Selling_Price'])

# DataFrame 2: Standardization
standard_scaler = StandardScaler()
train_data_standard = standard_scaler.fit_transform(train_data.drop('Selling_Price', axis=1))
test_data_standard = standard_scaler.transform(test_data.drop('Selling_Price', axis=1))

# Train and evaluate model with Standardization data
print("\nResults for Standardization Scaling:")
train_and_evaluate_model(train_data_standard, test_data_standard, train_data['Selling_Price'], test_data['Selling_Price'])

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(train_data_standard, train_data['Selling_Price'])
lasso_pred = lasso_model.predict(test_data_standard)

print("\nResults for Lasso Regression:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], lasso_pred))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], lasso_pred))

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(train_data_standard, train_data['Selling_Price'])
ridge_pred = ridge_model.predict(test_data_standard)

print("\nResults for Ridge Regression:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], ridge_pred))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], ridge_pred))

# Correlation matrix calculation and visualization
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatter plot between a feature and target
plt.scatter(df['Kms_Driven'], df['Selling_Price'])
plt.xlabel("Kms Driven")
plt.ylabel("Selling Price")
plt.title("Correlation between Kms Driven and Selling Price")
plt.show()

# Display correlation of each feature with target
all_features_correlation = df.corr()['Selling_Price']
print("Correlation between Selling_Price and other features:")
print(all_features_correlation)
