from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = fetch_california_housing()

# Convert to DataFrame for easier viewing
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseValue"] = data.target  # target = house price

# Show first 5 rows
print(df.head())

print(df.describe())

print(df.columns)



# ---- FEATURE DISTRIBUTIONS ----
plt.figure(figsize=(10,6))
sns.histplot(df['MedHouseValue'], kde=True)
plt.title("Distribution of House Prices (Median House Value)")
plt.show(block=False)
plt.pause(5)   # show it for 2 seconds
plt.close()    # close automatically


plt.figure(figsize=(10,6))
sns.histplot(df['MedInc'], kde=True)
plt.title("Distribution of Median Income")
plt.show(block=False)
plt.pause(5)   
plt.close()    




#step 3 prepare data for ML

X = df.drop("MedHouseValue", axis=1)
Y = df["MedHouseValue"]  #the price column 

#spliting the data set into trainig and test set
from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#training the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Predict on test data
Y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

#testinng/ comparing with the lasso model
lasso = Lasso(alpha=1.0, random_state=42)  
lasso.fit(X_train_scaled, Y_train)

Y_pred_lasso = lasso.predict(X_test_scaled)
mse_lasso = mean_squared_error(Y_test, Y_pred_lasso)
r2_lasso = r2_score(Y_test, Y_pred_lasso)

print("Lasso Regression Results:")
print("Mean Squared Error (MSE):", mse_lasso)
print("R² Score:", r2_lasso)

#testing/comparing with ridge model
ridge = Ridge(alpha=1.0, random_state=42)   
ridge.fit(X_train_scaled, Y_train)

Y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(Y_test, Y_pred_ridge)
r2_ridge = r2_score(Y_test, Y_pred_ridge)

print("\nRidge Regression Results:")
print("Mean Squared Error (MSE):", mse_ridge)
print("R² Score:", r2_ridge)
