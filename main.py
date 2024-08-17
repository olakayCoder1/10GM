import pandas as pd
import numpy as np
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
# print(dg.head())
# Filter for the specific user
user_id = '269-54-1394'
user_record = df[df['user_id'] == user_id]

# print(df.head())
total_amount = df['amt'].sum()


print(f"Total amount: {'{:,}'.format(total_amount)}")

total_records = len(user_record)
print(f"Total number of records:  {'{:,}'.format(total_records)}")

# Extract month and aggregate data
user_record['transaction_date'] = pd.to_datetime(user_record['test_created_at'])
user_record['month'] = user_record['transaction_date'].dt.to_period('M')




# Grouping data by month
monthly_data = user_record.groupby('month').agg({'amt': ['mean', 'sum'], 'transaction_id': 'nunique'}).reset_index()
monthly_data.columns = ['month', 'average_monthly_spend', 'total_spend', 'purchase_frequency']


# print(monthly_data)
# Calculating features
average_monthly_spend = monthly_data['average_monthly_spend'].mean()
total_spend = monthly_data['total_spend'].sum()
purchase_frequency = monthly_data['purchase_frequency'].sum()


print(f"average_monthly_spend : {'{:,}'.format(round(average_monthly_spend,2))}")
print(f"total_spend : {'{:,}'.format(round(total_spend,2))}")
print(f"purchase_frequency : {'{:,}'.format(round(purchase_frequency,2))}")

# Rule-based loan eligibility
spend_threshold = 3000
frequency_threshold = 50
is_eligible = (average_monthly_spend > spend_threshold and purchase_frequency > frequency_threshold)

print(f"Loan Eligibility (Rule-based): {'Eligible' if is_eligible else 'Not Eligible'}")


# Assuming eligibility is based on some threshold logic
if is_eligible:
    # Assuming you want to determine eligible loan amount based on average monthly spend
    eligible_loan_amount = average_monthly_spend * 10  # This multiplier is arbitrary and can be adjusted
    eligible_loan_amount = "{:,}".format(round(eligible_loan_amount,2))
    print(f"Eligible Loan Amount: ${eligible_loan_amount}")
    # print(f"Eligible Loan Amount: ${eligible_loan_amount:.2f}")
else:
    print("Eligible Loan Amount: Not Eligible")
