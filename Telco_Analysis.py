import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
df = pd.read_csv("telco.csv")
df.head()
pd.set_option("display.max_column", None)
pd.set_option("display.width", 1000)
df.info()
df.dtypes
from sklearn.impute import SimpleImputer
#chacking Missing values
df.isnull().sum().sum()

# Columns to impute
columns_to_impute = ["Offer", "Internet Type", "Churn Category", "Churn Reason"]

# Initialize the imputer with the chosen strategy
imputer = SimpleImputer(strategy="most_frequent")

# Fit and transform each column in one step
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

df.duplicated().sum()
df.describe()
# Setting up to verify observations, starting with Customer Status distribution
## Count plot for Customer Status
plt.figure(figsize=(5,5))
ax = sns.countplot(x = "Customer Status", data = df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customer Status")
plt.show()

## Pie chart for Customer Status percentages
plt.figure(figsize=(5,5))
gb = df.groupby("Customer Status").agg ({"Customer Status": "count"})
plt.pie(gb["Customer Status"], labels=gb.index, autopct="%1.2f%%")
plt.title("Percentage of Customer Status")
plt.show()
#26.54% customers are churned

## Gender and Churn relationship
plt.figure(figsize=(5,5))
ax = sns.countplot(x = "Gender", data = df, hue= "Customer Status")
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.title("CCount of Gender by Customer Status")
plt.show()
#There is no major differences between churn customer status and the gender

# Senior Citizen and Churn relationship
plt.figure(figsize=(5,5))
ax = sns.countplot(x = "Senior Citizen", data = df, hue= "Customer Status")
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.title("Count of Senior Citizen by Customer Status")
plt.show()

# Recalculating to ensure the data for stacked bar with active customer percentage is clear
# Getting the distribution for each category

# Calculating percentage distribution for each Senior Citizen category by Customer Status
count_data = df.groupby(['Senior Citizen', 'Customer Status']).size().unstack().fillna(0)
count_data_percent = count_data.div(count_data.sum(axis=1), axis=0) * 100

# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the data
count_data_percent.plot(kind='bar', stacked=True, ax=ax)
ax.set_title("Percentage of Customer Status by Senior Citizen")
ax.set_xlabel("Senior Citizen")
ax.set_ylabel("Percentage (%)")

# Adding percentage labels on the bars, highlighting the "Active" status percentage
for i, container in enumerate(ax.containers):
    status_label = "Churn" if i == 0 else "Not-Churn"
    for rect, percentage in zip(container, container.datavalues):
        label_text = f"{status_label} {percentage:.1f}%"
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                label_text, ha="center", va="center", color="black")

plt.show()
#Senior Citizen customers are more churned then the Stayed and resently joined customers
df.head()

# Tenure and Churn relationship
plt.figure(figsize = (10,6))
sns.histplot(x = "Tenure in Months", data= df, bins= 72, hue = "Customer Status")
plt.show()
#people who have used our services for a long time have stayed and people who have used our
# services for one or two months have churn

# Contract Type and Churn
plt.figure(figsize=(5,5))
ax = sns.countplot(x = "Contract", data = df, hue= "Customer Status")
ax.bar_label(ax.containers[0])
plt.title("Tenure in Months by Customer Status")
plt.show()
#from this data it was cleared that month-to-month customers are more churned then the logn times customers
df.columns.values

## Offer and Churn relationship
plt.figure(figsize=(10,5))
ax = sns.countplot(x = "Offer", data = df, hue= "Customer Status")
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.title("Customer Status by Offer")
plt.show()

# Calculating percentage distribution for each Offer by Customer Status
offer_data = df.groupby(['Offer', 'Customer Status']).size().unstack().fillna(0)
offer_data_percent = offer_data.div(offer_data.sum(axis=1), axis=0) * 100

# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the data
offer_data_percent.plot(kind='bar', stacked=True, ax=ax)
ax.set_title("Percentage of Customer Status by Offer")
ax.set_xlabel("Offer")
ax.set_ylabel("Percentage (%)")

# Adding percentage labels on the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")

plt.show()
#from this data almost 52.91% customers are churned under offer E plan
# Defining the columns for categorical counts
categorical_columns = [
    'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
    'Online Security', 'Online Backup', 'Device Protection Plan',
    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
    'Streaming Music', 'Unlimited Data'
]

# Setting up the subplots for each categorical column
num_cols = 4
num_rows = (len(categorical_columns) + num_cols - 1) // num_cols  # Calculate rows needed for subplot grid

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))

# Loop over categorical columns and plot count plots in subplots
for i, col in enumerate(categorical_columns):
    ax = axes[i // num_cols, i % num_cols]  # Determine correct subplot position
    sns.countplot(x=col, data=df, ax=ax, hue = df["Customer Status"])
    ax.set_title(f'Count of {col}')
    ax.set_xlabel('')
    ax.set_ylabel('Count')

# Hide any empty subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes[j // num_cols, j % num_cols])

plt.tight_layout()
plt.show()
df.head()
#These visualizations suggest that certain add-on services and 
# features may play a significant role in retaining customers. 
# Services like online security, tech support, and device protection seem to be positively 
# associated with customer retention, whereas customers without these services or with "Fiber Optic" 
# internet are at higher risk of churn.

## Payment Method and Churn relationship
plt.figure(figsize=(10,5))
ax = sns.countplot(x = "Payment Method", data = df, hue= "Customer Status")
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.title("Customer Status by Payment Method")
plt.show()
#Customers who payments through Bank Withdrawal