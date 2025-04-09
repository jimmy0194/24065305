# 24065305.py
# Fundamentals of Data Science - LUCKY - Student ID: 24065305

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df_24065305 = pd.read_csv('sales0.csv')
df_24065305['Date'] = pd.to_datetime(df_24065305['Date'], errors='coerce')
df_24065305 = df_24065305.dropna(subset=['Date'])

# Extract time-based features
df_24065305['Year'] = df_24065305['Date'].dt.year
df_24065305['Month'] = df_24065305['Date'].dt.month
df_24065305['DayOfYear'] = df_24065305['Date'].dt.dayofyear

# Calculate total items sold and revenue
df_24065305['TotalItemsSold'] = (
    df_24065305['NumberGroceryShop'] + df_24065305['NumberGroceryOnline'] +
    df_24065305['NumberNongroceryShop'] + df_24065305['NumberNongroceryOnline']
)
df_24065305['RevenueGrocery'] = (
    df_24065305['NumberGroceryShop'] * df_24065305['PriceGroceryShop'] +
    df_24065305['NumberGroceryOnline'] * df_24065305['PriceGroceryOnline']
)
df_24065305['RevenueNongrocery'] = (
    df_24065305['NumberNongroceryShop'] * df_24065305['PriceNongroceryShop'] +
    df_24065305['NumberNongroceryOnline'] * df_24065305['PriceNongroceryOnline']
)
df_24065305['TotalRevenue'] = df_24065305['RevenueGrocery'] + df_24065305['RevenueNongrocery']
df_24065305['AveragePrice'] = df_24065305['TotalRevenue'] / df_24065305['TotalItemsSold']

# === Step 2: Monthly Avg Daily Items Sold (Bar Chart) + Fourier ===
monthly_avg_24065305 = df_24065305.groupby('Month')['TotalItemsSold'].mean()
df_2022 = df_24065305[df_24065305['Year'] == 2022]
y_2022 = df_2022['TotalItemsSold'].values
x_2022 = np.arange(1, len(y_2022) + 1)

N = len(y_2022)
a0 = np.mean(y_2022)
fourier_series = np.full(N, a0 / 2)

for n in range(1, 9):
    an = 2 / N * np.sum(y_2022 * np.cos(2 * np.pi * n * x_2022 / N))
    bn = 2 / N * np.sum(y_2022 * np.sin(2 * np.pi * n * x_2022 / N))
    fourier_series += an * np.cos(2 * np.pi * n * x_2022 / N) + bn * np.sin(2 * np.pi * n * x_2022 / N)

plt.figure(figsize=(14, 6))
plt.bar(monthly_avg_24065305.index, monthly_avg_24065305.values, color='mediumseagreen', label='Monthly Avg Items Sold')
plt.plot(np.linspace(1, 12, N), fourier_series, color='purple', label='Fourier Approx (8 terms)')
plt.title('Figure 1 - Monthly Avg Items Sold + Fourier (ID: 24065305)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Avg Items Sold per Day')
plt.xticks(ticks=np.arange(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.legend()
plt.text(11, max(monthly_avg_24065305) + 100, 'Student ID: 24065305', fontsize=12)
plt.tight_layout()
plt.savefig('figure1_24065305.png')
plt.close()

# === Step 3 + 4 + 5: Scatter Plot + Linear Regression + X/Y ===
plt.figure(figsize=(10, 6))
plt.scatter(df_24065305['TotalItemsSold'], df_24065305['AveragePrice'], alpha=0.6, label='Daily Data')
plt.xlabel('Items Sold')
plt.ylabel('Average Price')
plt.title('Figure 2 - Avg Price vs Items Sold with Regression (ID: 24065305)', fontsize=14)

# Linear regression
X = df_24065305['TotalItemsSold'].values.reshape(-1, 1)
y = df_24065305['AveragePrice'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
plt.plot(df_24065305['TotalItemsSold'], y_pred, color='red', label='Linear Regression')

# Calculate X and Y (max/min avg price day number)
avg_price_by_day = df_24065305.groupby('DayOfYear')['AveragePrice'].mean()
X_value = avg_price_by_day.idxmax()
Y_value = avg_price_by_day.idxmin()

plt.text(df_24065305['TotalItemsSold'].min(), df_24065305['AveragePrice'].max(),
         f'X (Highest Avg Price Day): {X_value}', fontsize=10)
plt.text(df_24065305['TotalItemsSold'].min(), df_24065305['AveragePrice'].max() - 0.5,
         f'Y (Lowest Avg Price Day): {Y_value}', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('figure2_24065305.png')
plt.close()

# === Step 6: Print X and Y ===
print("Student ID: 24065305")
print(f"X (Highest Avg Price Day): {X_value}")
print(f"Y (Lowest Avg Price Day): {Y_value}")