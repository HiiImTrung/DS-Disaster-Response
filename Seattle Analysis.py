#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# <h2 id='part1'>A Look at the Data</h2>
# 
# Using the Seattle Airbnb (https://www.kaggle.com/datasets/airbnb/seattle/data) dataset from Kaggle, we would like to deep dive and discuss these three business-oriented questions:
# 
# 1. What are the seasonal trends in pricing across Seattle?
# 2. What factors impact a listing's average rating or occupancy?
# 3. Are there certain neighborhoods or property types with the highest bookings?
# 
# <h2 id='part2'>About the Dataset</h2>
# 
# <h3 id='Context'>Context</h3>
# 
# Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. As part of the Airbnb Inside initiative, this dataset describes the listing activity of homestays in Seattle, WA.
# 
# <h3 id='Content'>Content</h3>
# The following Airbnb activity is included in this Seattle dataset:
# 
# Listings, including full descriptions and average review score
# Reviews, including unique id for each reviewer and detailed comments
# Calendar, including listing id and the price and availability for that day
# 
# 

# In[1]:



import pandas as pd

# Load datasets
listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')
calendar = pd.read_csv('calendar.csv')

# View the first few rows of each dataset
display(listings.head())
display(reviews.head())
display(calendar.head())


# In[2]:


print(f'Listings shape: {listings.shape}')
print(f'Reviews shape: {reviews.shape}')
print(f'Calendar shape: {calendar.shape}')

# Check for data types and null values
listings.info()
reviews.info()
calendar.info()


# In[3]:


# Remove dollar sign and convert `price` to numeric
calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)

# Check for missing values in `calendar`
calendar.isna().sum()


# In[5]:


# Count the number of reviews per listing
review_counts = reviews.groupby('listing_id').size().reset_index(name='review_count')
listings = listings.merge(review_counts, left_on='id', right_on='listing_id', how='left')
listings['review_count'] = listings['review_count'].fillna(0)


# In[7]:


# Fill or drop missing values as needed
listings['review_scores_value'].fillna(listings['review_scores_value'].median(), inplace=True)
listings['neighborhood_overview'].fillna('No overview', inplace=True)


# In[8]:


# Convert `date` in `calendar.csv` to datetime
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['month'] = calendar['date'].dt.month

# Average price per month
monthly_price_trend = calendar.groupby('month')['price'].mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
monthly_price_trend.plot(kind='line', marker='o')
plt.title('Average Monthly Price Trend')
plt.xlabel('Month')
plt.ylabel('Average Price ($)')
plt.grid(True)
plt.show()


# In[9]:


import seaborn as sns

# Correlation analysis for numerical fields
sns.heatmap(listings[['review_scores_value', 'price', 'review_count']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Review Score and Other Factors')
plt.show()

# Scatter plot for price vs. review scores
plt.figure(figsize=(10, 5))
sns.scatterplot(data=listings, x='price', y='review_scores_value')
plt.title('Price vs Review Score')
plt.xlabel('Price ($)')
plt.ylabel('Review Score')
plt.grid(True)
plt.show()


# In[10]:


# Group by neighborhood and calculate average price and review count
neighborhood_stats = listings.groupby('neighbourhood_cleansed').agg({
    'price': 'mean',
    'review_count': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=neighborhood_stats.sort_values(by='review_count', ascending=False).head(10), x='review_count', y='neighbourhood_cleansed')
plt.title('Top 10 Neighborhoods by Average Number of Reviews')
plt.xlabel('Average Number of Reviews')
plt.ylabel('Neighborhood')
plt.grid(True)
plt.show()


# In[11]:


## Key Insights and Conclusions
1. **Seasonal Price Trends**: Prices in Seattle tend to peak in the summer months, suggesting high demand in warmer weather.
2. **Factors Affecting Ratings**: High prices slightly correlate with lower ratings, potentially due to raised expectations or value perceptions.
3. **Popular Neighborhoods**: Listings in areas such as Capitol Hill and Downtown Seattle have higher review counts, indicating popularity.

