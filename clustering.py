import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv('data1.csv')

# Define the range for the past six months
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

# Function to generate a random date within the last six months
def generate_random_date(start, end):
    return start + timedelta(days=np.random.randint(0, (end - start).days))

# Apply the function to generate random dates
df['test_created_at'] = df.apply(lambda _: generate_random_date(start_date, end_date), axis=1)

df = df[['user_id','amt','transaction_id','test_created_at']]
user_id = '269-54-1394'
user_record = df[df['user_id'] == user_id]


# for r in df['amt']:
#     print(r)
# Extract month and aggregate data
user_record['transaction_date'] = pd.to_datetime(user_record['test_created_at'])
user_record['month'] = user_record['transaction_date'].dt.to_period('M')

# Grouping data by month
monthly_data = user_record.groupby('month').agg({'amt': ['mean', 'sum'], 'transaction_id': 'nunique'}).reset_index()
monthly_data.columns = ['month', 'average_monthly_spend', 'total_spend', 'purchase_frequency']

# Calculating features
average_monthly_spend = monthly_data['average_monthly_spend'].mean()
total_spend = monthly_data['total_spend'].sum()
purchase_frequency = monthly_data['purchase_frequency'].sum()

# Preparing the data for modeling
historical_data = df[df['user_id'] != user_id]  # Use all other users' data for training
historical_data_grouped = historical_data.groupby('user_id').agg({
    'amt': 'mean',
    'transaction_id': 'nunique'
}).reset_index()
historical_data_grouped.columns = ['user_id', 'average_monthly_spend', 'purchase_frequency']

# Feature scaling
scaler = StandardScaler()
X = historical_data_grouped[['average_monthly_spend', 'purchase_frequency']]
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
historical_data_grouped['cluster'] = clusters

# Train-test split for regression
y = historical_data_grouped['average_monthly_spend'] * 10  # Assuming loan amount is 10 times the average spend
print(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
regression_model = RandomForestRegressor(random_state=42)
regression_model.fit(X_train, y_train)

# Evaluate model
y_pred_reg = regression_model.predict(X_test)
regression_mse = mean_squared_error(y_test, y_pred_reg)
print(f"Regression Model MSE: {regression_mse:.2f}")

# Predict eligible loan amount for the specific user
new_user_features = pd.DataFrame({
    'average_monthly_spend': [average_monthly_spend],
    'purchase_frequency': [purchase_frequency]
})
new_user_features_scaled = scaler.transform(new_user_features)
predicted_loan_amount = regression_model.predict(new_user_features_scaled)[0]

print(f"Predicted Loan Amount: ${predicted_loan_amount:,.2f}")
