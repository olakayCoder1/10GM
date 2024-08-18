import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib



class UserModelManager:
    def __init__(self, data_path, models_path):
        self.data_path = data_path
        self.models_path = models_path
        self.df = pd.read_csv(data_path)
        self.ensure_models_directory_exists()

    def ensure_models_directory_exists(self):
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def generate_random_date(self, start, end):
        return start + timedelta(days=np.random.randint(0, (end - start).days))

    def preprocess_data(self):
        # Define the range for the past six months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        # Apply the function to generate random dates
        self.df['test_created_at'] = self.df.apply(lambda _: self.generate_random_date(start_date, end_date), axis=1)
        self.df = self.df[['user_id', 'amt', 'transaction_id', 'test_created_at']]

    def user_record(self, user_id):
        user_record = self.df[self.df['user_id'] == user_id]
        return user_record

    def calculate_features(self, user_record):
        user_record['transaction_date'] = pd.to_datetime(user_record['test_created_at'])
        user_record['month'] = user_record['transaction_date'].dt.to_period('M')

        # Grouping data by month
        monthly_data = user_record.groupby('month').agg({'amt': ['mean', 'sum'], 'transaction_id': 'nunique'}).reset_index()
        monthly_data.columns = ['month', 'average_monthly_spend', 'total_spend', 'purchase_frequency']

        # Calculating features
        average_monthly_spend = monthly_data['average_monthly_spend'].mean()
        total_spend = monthly_data['total_spend'].sum()
        purchase_frequency = monthly_data['purchase_frequency'].sum()
        
        return average_monthly_spend, total_spend, purchase_frequency

    def train_and_save_model(self, user_id, user_record):
        # Prepare historical data
        historical_data = self.df[self.df['user_id'] != user_id]
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
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train regression model
        regression_model = RandomForestRegressor(random_state=42)
        regression_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred_reg = regression_model.predict(X_test)
        regression_mse = mean_squared_error(y_test, y_pred_reg)
        print(f"Regression Model MSE for user {user_id}: {regression_mse:.2f}")

        # Save the models
        user_model_path = os.path.join(self.models_path, user_id)
        if not os.path.exists(user_model_path):
            os.makedirs(user_model_path)

        joblib.dump(regression_model, os.path.join(user_model_path, 'regression_model.pkl'))
        joblib.dump(scaler, os.path.join(user_model_path, 'scaler.pkl'))
        joblib.dump(kmeans, os.path.join(user_model_path, 'kmeans.pkl'))

    def load_model(self, user_id):
        user_model_path = os.path.join(self.models_path, user_id)
        if os.path.exists(user_model_path):
            regression_model = joblib.load(os.path.join(user_model_path, 'regression_model.pkl'))
            scaler = joblib.load(os.path.join(user_model_path, 'scaler.pkl'))
            kmeans = joblib.load(os.path.join(user_model_path, 'kmeans.pkl'))
            return regression_model, scaler, kmeans
        else:
            return None, None, None

    def predict_for_user(self, user_id):
        user_record = self.user_record(user_id)
        if user_record.empty:
            print(f"No records found for user {user_id}.")
            return None

        average_monthly_spend, _, purchase_frequency = self.calculate_features(user_record)
        
        regression_model, scaler, kmeans = self.load_model(user_id)
        
        if regression_model is None:
            print(f"Model for user {user_id} not found. Training a new model.")
            self.train_and_save_model(user_id, user_record)
            regression_model, scaler, kmeans = self.load_model(user_id)
        
        # Prepare new user features
        new_user_features = pd.DataFrame({
            'average_monthly_spend': [average_monthly_spend],
            'purchase_frequency': [purchase_frequency]
        })
        new_user_features_scaled = scaler.transform(new_user_features)

        # Predict loan amount
        predicted_loan_amount = regression_model.predict(new_user_features_scaled)[0]
        return predicted_loan_amount



if __name__ == "__main__":
    data_path = 'data1.csv'
    models_path = 'models'
    user_id = '197-71-6160'

    manager = UserModelManager(data_path, models_path)
    manager.preprocess_data()

    predicted_loan_amount = manager.predict_for_user(user_id)
    if predicted_loan_amount is not None:
        print(f"Predicted Loan Amount for User {user_id}: ${predicted_loan_amount:,.2f}")
