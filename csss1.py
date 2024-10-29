
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# خواندن داده
df = pd.read_csv('cardata.csv')

# تبدیل مقادیر رشته‌ای به عدد
label_encoder = LabelEncoder()
df['Car_Name'] = label_encoder.fit_transform(df['Car_Name'])
df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = label_encoder.fit_transform(df['Seller_Type'])
df['Transmission'] = label_encoder.fit_transform(df['Transmission'])

# تقسیم داده به دو قسمت آموزش و تست
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# دیتافریم اول: مقیاس بندی Min-Max
min_max_scaler = MinMaxScaler()
train_data_minmax = min_max_scaler.fit_transform(train_data.drop('Selling_Price', axis=1))
test_data_minmax = min_max_scaler.transform(test_data.drop('Selling_Price', axis=1))


# تابع برای آموزش مدل و نمایش معیارهای ارزیابی
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # آموزش مدل
    model = LinearRegression()
    model.fit(X_train, y_train)

    # پیش‌بینی مقادیر
    y_pred = model.predict(X_test)

    # معیارهای ارزیابی
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # نمایش نتایج
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)

    # نمایش نمودار پیش‌بینی در مقابل واقعی
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Selling Price")
    plt.show()


# آموزش و ارزیابی مدل با داده مقیاس بندی Min-Max
print("Results for Min-Max Scaling:")
train_and_evaluate_model(train_data_minmax, test_data_minmax, train_data['Selling_Price'], test_data['Selling_Price'])

# دیتافریم دوم: مقیاس بندی معیارگذاری
standard_scaler = StandardScaler()
train_data_standard = standard_scaler.fit_transform(train_data.drop('Selling_Price', axis=1))
test_data_standard = standard_scaler.transform(test_data.drop('Selling_Price', axis=1))


# آموزش و ارزیابی مدل با داده مقیاس بندی معیارگذاری
print("Results for Standardization Scaling:")
train_and_evaluate_model(train_data_standard, test_data_standard, train_data['Selling_Price'],
                         test_data['Selling_Price'])


# آموزش و ارزیابی مدل با Lasso Regression
lasso_model = Lasso(alpha=0.1)  # مقدار alpha را تنظیم کنید
lasso_model.fit(train_data_standard, train_data['Selling_Price'])
lasso_pred = lasso_model.predict(test_data_standard)

print("\nResults for Lasso Regression:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], lasso_pred))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], lasso_pred))

# آموزش و ارزیابی مدل با Ridge Regression
ridge_model = Ridge(alpha=1.0)  # مقدار alpha را تنظیم کنید
ridge_model.fit(train_data_standard, train_data['Selling_Price'])
ridge_pred = ridge_model.predict(test_data_standard)

print("\nResults for Ridge Regression:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], ridge_pred))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], ridge_pred))



# محاسبه ماتریس کورلیشن
correlation_matrix = df.corr()

# نمایش ماتریس کورلیشن با استفاده از heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# رسم نمودار کورلیشن بین یک فیچر و تارگت
plt.scatter(df['Kms_Driven'], df['Selling_Price'])
plt.xlabel("Kms Driven")
plt.ylabel("Selling Price")
plt.title("Correlation between Kms Driven and Selling Price")
plt.show()


# محاسبه ماتریس کورلیشن بین تمام فیچرها و تارگت
all_features_correlation = df.corr()['Selling_Price']

# نمایش ماتریس کورلیشن با استفاده از heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (All Features)")
plt.show()

# نمایش کورلیشن هر فیچر با تارگت
print("Correlation between Selling_Price and other features:")
print(all_features_correlation)


# آموزش و ارزیابی مدل با داده مقیاس بندی Min-Max
print("Results for Min-Max Scaling:")
train_and_evaluate_model(train_data_minmax, test_data_minmax, train_data['Selling_Price'], test_data['Selling_Price'])

# آموزش و ارزیابی مدل با داده مقیاس بندی معیارگذاری
print("\nResults for Standardization Scaling:")
train_and_evaluate_model(train_data_standard, test_data_standard, train_data['Selling_Price'], test_data['Selling_Price'])

# آموزش و ارزیابی مدل با Lasso Regression برای مقیاس بندی Min-Max
lasso_model_minmax = Lasso(alpha=0.1)  # مقدار alpha را تنظیم کنید
lasso_model_minmax.fit(train_data_minmax, train_data['Selling_Price'])
lasso_pred_minmax = lasso_model_minmax.predict(test_data_minmax)

print("\nResults for Lasso Regression with Min-Max Scaling:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], lasso_pred_minmax))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred_minmax))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred_minmax, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], lasso_pred_minmax))

# آموزش و ارزیابی مدل با Ridge Regression برای مقیاس بندی Min-Max
ridge_model_minmax = Ridge(alpha=1.0)  # مقدار alpha را تنظیم کنید
ridge_model_minmax.fit(train_data_minmax, train_data['Selling_Price'])
ridge_pred_minmax = ridge_model_minmax.predict(test_data_minmax)

print("\nResults for Ridge Regression with Min-Max Scaling:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], ridge_pred_minmax))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred_minmax))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred_minmax, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], ridge_pred_minmax))

# آموزش و ارزیابی مدل با Lasso Regression برای مقیاس بندی معیارگذاری
lasso_model_standard = Lasso(alpha=0.1)  # مقدار alpha را تنظیم کنید
lasso_model_standard.fit(train_data_standard, train_data['Selling_Price'])
lasso_pred_standard = lasso_model_standard.predict(test_data_standard)

print("\nResults for Lasso Regression with Standardization Scaling:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], lasso_pred_standard))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred_standard))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], lasso_pred_standard, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], lasso_pred_standard))

# آموزش و ارزیابی مدل با Ridge Regression برای مقیاس بندی معیارگذاری
ridge_model_standard = Ridge(alpha=1.0)  # مقدار alpha را تنظیم کنید
ridge_model_standard.fit(train_data_standard, train_data['Selling_Price'])
ridge_pred_standard = ridge_model_standard.predict(test_data_standard)

print("\nResults for Ridge Regression with Standardization Scaling:")
print("Mean Absolute Error:", mean_absolute_error(test_data['Selling_Price'], ridge_pred_standard))
print("Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred_standard))
print("Root Mean Squared Error:", mean_squared_error(test_data['Selling_Price'], ridge_pred_standard, squared=False))
print("R2 Score:", r2_score(test_data['Selling_Price'], ridge_pred_standard))
