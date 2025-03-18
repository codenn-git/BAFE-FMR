#validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv(r"C:\Users\user-307E4B3400\Desktop\BAFE FMR\NUEVA ECIJA FMR\validation\fieldxderived.csv")
df = df[df['derived'] <= 8]

# Scatter plot with equal axis intervals and x-axis range set to min and max values of the data
plt.scatter(df['field'], df['derived'])
plt.xlabel('Field')
plt.ylabel('Derived')
plt.title('Validation')
plt.xlim(4.9, 5.2) #plt.xlim(df['field'].min(), df['field'].max())
# plt.ylim(df['field'].min(), df['field'].max())

# Compute the linear regression
lr_model = LinearRegression()
X_lr = df['field'].values.reshape(-1, 1)
y = df['derived'].values
lr_model.fit(X_lr, y)

# Predict using the linear regression model
y_lr_pred = lr_model.predict(X_lr)

# Calculate the R-squared value
r2 = r2_score(y, y_lr_pred)
print(f'R-squared: {r2}')

# Plot the linear regression line
plt.plot(df['field'], y_lr_pred, color='blue', label='Linear Regression')

# Add the equation of the trend line
slope = lr_model.coef_[0]
intercept = lr_model.intercept_
equation = f'y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.legend()
plt.show()