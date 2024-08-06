import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Load CSV files from the current directory
market_trend_data = pd.read_csv('./file_CSV/MarketTrendData.csv')
product_detail_data = pd.read_csv('./file_CSV/ProductDetailData.csv')
website_access_category_data = pd.read_csv('./file_CSV/WebsiteAccessCategoryData.csv')
product_group_data = pd.read_csv('./file_CSV/ProductGroupData.csv')
sale_data = pd.read_csv('./file_CSV/SaleData.csv')
customer_data = pd.read_csv('./file_CSV/CustomerData.csv')

# Display basic information about the data
print("Market Trend Data:")
print(market_trend_data.info())
print("\nProduct Detail Data:")
print(product_detail_data.info())
print("\nWebsite Access Category Data:")
print(website_access_category_data.info())
print("\nProduct Group Data:")
print(product_group_data.info())
print("\nSale Data:")
print(sale_data.info())
print("\nCustomer Data:")
print(customer_data.info())

# Define a function to clean the data
def clean_data(df):
    # Drop columns or rows that are completely empty
    df.dropna(how='all', inplace=True)
    # Fill missing values with mean for numeric columns and mode for object columns
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

# Clean each DataFrame using the clean_data function
market_trend_data = clean_data(market_trend_data)
product_detail_data = clean_data(product_detail_data)
website_access_category_data = clean_data(website_access_category_data)
product_group_data = clean_data(product_group_data)
sale_data = clean_data(sale_data)
customer_data = clean_data(customer_data)

# Display information after cleaning the data
print("\nMarket Trend Data after cleaning:")
print(market_trend_data.info())
print("\nProduct Detail Data after cleaning:")
print(product_detail_data.info())
print("\nWebsite Access Category Data after cleaning:")
print(website_access_category_data.info())
print("\nProduct Group Data after cleaning:")
print(product_group_data.info())
print("\nSale Data after cleaning:")
print(sale_data.info())
print("\nCustomer Data after cleaning:")
print(customer_data.info())

# Save the cleaned DataFrames to new CSV files
market_trend_data.to_csv('./file_CSV/cleaned_MarketTrendData.csv', index=False)
product_detail_data.to_csv('./file_CSV/cleaned_ProductDetailData.csv', index=False)
website_access_category_data.to_csv('./file_CSV/cleaned_WebsiteAccessCategoryData.csv', index=False)
product_group_data.to_csv('./file_CSV/cleaned_ProductGroupData.csv', index=False)
sale_data.to_csv('./file_CSV/cleaned_SaleData.csv', index=False)
customer_data.to_csv('./file_CSV/cleaned_CustomerData.csv', index=False)

# Market Trend Data Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(data=market_trend_data, x='StartDate', y='ImpactScore')
plt.title('Market Trend Over Time')
plt.xlabel('Start Date')
plt.ylabel('Impact Score')
plt.xticks(rotation=45)
plt.show()

# Product Detail Data Visualization
plt.figure(figsize=(12, 8))
sns.barplot(data=product_detail_data, x='CategoryID', y='UnitPrice')
plt.title('Product Sales by Category')
plt.xlabel('Category ID')
plt.ylabel('Unit Price')
plt.xticks(rotation=45)
plt.show()

# Website Access Category Data Visualization
website_access_category_counts = website_access_category_data['PageViewed'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(website_access_category_counts, labels=website_access_category_counts.index, autopct='%1.1f%%')
plt.title('Website Access Categories')
plt.show()

# Correlation Heatmap for Product Group Data
plt.figure(figsize=(10, 8))
# Filter out only numeric columns for correlation
numeric_columns = product_group_data.select_dtypes(include=['number'])
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Product Group Data')
plt.show()

# Sale Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(sale_data['TotalPrice'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Total Price')
plt.ylabel('Frequency')
plt.show()

# Customer Data Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='PostalCode', y='CustomerID')
plt.title('Customer Postal Code vs. Customer ID')
plt.xlabel('Postal Code')
plt.ylabel('Customer ID')
plt.show()

# Merge sales data with customer data based on CustomerID
merged_data = pd.merge(sale_data, customer_data, on='CustomerID')

# Aggregate sales data by customer
customer_sales_aggregation = merged_data.groupby('CustomerID').agg(
    TotalSales=('TotalPrice', 'sum'),
    AverageSales=('TotalPrice', 'mean'),
    TransactionCount=('SaleID', 'count')
).reset_index()

# Select independent variables and dependent variable
X = customer_sales_aggregation[['AverageSales', 'TransactionCount']]
y = customer_sales_aggregation['TotalSales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the regression coefficients
print('Coefficients:', linear_model.coef_)
print('Intercept:', linear_model.intercept_)

# Select features for clustering
X_clustering = customer_sales_aggregation[['AverageSales', 'TransactionCount', 'TotalSales']]

# Initialize and train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
customer_sales_aggregation['Cluster'] = kmeans.fit_predict(X_clustering)

# Visualize the clusters
plt.scatter(customer_sales_aggregation['AverageSales'], customer_sales_aggregation['TotalSales'], 
            c=customer_sales_aggregation['Cluster'])
plt.xlabel('Average Sales')
plt.ylabel('Total Sales')
plt.title('Customer Segmentation')
plt.show()

# Load sales data from a CSV file
sales_data = pd.read_csv('./file_CSV/SaleData.csv')

# Convert SaleDate to datetime format
sales_data['SaleDate'] = pd.to_datetime(sales_data['SaleDate'])

# Extract year, month, and day as separate features
sales_data['Year'] = sales_data['SaleDate'].dt.year
sales_data['Month'] = sales_data['SaleDate'].dt.month
sales_data['Day'] = sales_data['SaleDate'].dt.day

# Aggregate data to get monthly sales
monthly_sales = sales_data.groupby(['Year', 'Month']).agg(
    TotalSales=('TotalPrice', 'sum')).reset_index()

# Select features and target
X = monthly_sales[['Year', 'Month']]
y = monthly_sales['TotalSales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Generate future months for prediction (e.g., next 12 months)
future_months = pd.DataFrame({
    'Year': np.repeat(2024, 12),
    'Month': np.arange(1, 13)
})

# Predict future sales
future_sales_pred = model.predict(future_months)

# Create a DataFrame for future sales predictions
future_sales = future_months.copy()
future_sales['TotalSales'] = future_sales_pred

print("Future Sales Predictions:")
print(future_sales)

# Linear regression using statsmodels
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Plot with matplotlib
plt.plot(X['Year'], y)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Line Chart')
plt.show()

# Plot with seaborn
sns.lineplot(x='Year', y='TotalSales', data=monthly_sales)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Line Chart')
plt.show()

# Plot with plotly
fig = px.line(monthly_sales, x='Year', y='TotalSales', title='Line Chart')
fig.show()
