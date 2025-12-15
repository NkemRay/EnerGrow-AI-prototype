#!/usr/bin/env python
# coding: utf-8

# # Malaria Prevalence Analysis in Benue state
# 

# In[1]:


#importing the necessary packages

import pandas as pd 
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt



# In[2]:


#pip install xlrd


# In[3]:


#load malaria datasets

malaria = pd.read_excel(r"C:\Users\USER\Downloads\Benue_Malaria.xls")


# In[4]:


print(malaria.head())


# In[5]:


# loading the climate data 

climate = pd.read_excel(r"C:\Users\USER\Downloads\Climate Data.xlsx")
#printing the first few rows
print(climate.head())


# In[6]:


# merge on the year column

df = pd.merge(malaria, climate, on = "datetime", how = "inner")



# In[7]:


#printing the merged data
# To understand the number of rows and columns in the datasets
df.shape




# In[8]:


# checking for missing values

df.isna().sum()


# In[9]:


#checking for duplicate

df.duplicated().sum()


# In[10]:


#check for nulls and data types

df.info()


# In[11]:


# Summary statistics 
df.describe()


#Insights:
### approximately 1651 people reported fever per observation period, from as low as 85 up to 11,629 cases
### The maximum persons tested with RDT is equal to persons with fever suggesting sometimes all suspedted cases were tested.
### approximately 64% of those tested by RDT are positive (1000/1562) indicating persistent high transmission
### Diagnostic reliance is almost entirely on RDTs. Microscopy capacity is very limited in routine surveillance


# In[12]:


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# In[13]:


# Group by LGA and count number of cases
cases_per_lga = df.groupby("LGA")["positive_by RDT"].sum().reset_index()

# Rename columns for clarity
cases_per_lga.columns = ["LGA", "Number_of_Cases"]

# Display results
print(cases_per_lga)


# In[14]:


# Visualizing malaria prevalence distribution by climate factors
# Temperature
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='positive_by RDT',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='positive_by RDT',kde=True,hue='Temperature (celsuis)')
plt.show()

# Distribution of Rainfall

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='positive_by RDT',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='positive_by RDT',kde=True,hue='Rainfall (mm)')
plt.show()

# Distribution of malaria prevalence
plt.hist(df['positive_by RDT'], bins=30, edgecolor='black')
plt.xlabel('Malaria Cases (RDT positive)')
plt.ylabel('Frequency')
plt.title('Distribution of Malaria Cases')
plt.show()
# from the output below is it evident that malaria cases are highly skewed positively
#Most values are clustered at the lower end, with a long tail of large/extreme values


# In[15]:


# to calaculate the skewness of the data

print(df['Rainfall (mm)'].skew())

print(df['Temperature (celsuis)'].skew())

print(df['positive_by RDT'].skew())
#  A few periods/locations have very high malaria positives (possible outbreaks), while most are low.

## Using raw values in regression or time-series models may cause poor fit or biased coefficients because the distribution is not normal.


# In[16]:


# Normalization and reducing the skewness of the data
np.sqrt(df['positive_by RDT'])


# In[17]:


df['positive_by RDT'].skew()

## the output still shows that the data is highly dispersed and there can not conduct a regression analysis on it as it not mamally distributed
## Even after log transform, the data distribution is still pulled to the right with most values clustered on the lower end.


# In[18]:


# checking for the mean and varinace
mean_val = df['positive_by RDT'].mean()
var_val = df['positive_by RDT'].var()

print("Mean:", mean_val)
print("Variance:", var_val)
print("Variance-to-Mean Ratio:", var_val / mean_val)

### if the variance to mean ration is approximately 1 then poisson appropriate
### if the ratio >> 1 ( like 10, 100, 1000 then it shows strong overdispertion  negative binomial preffered


# In[19]:


# plotting malaria prevalence over time 
# Create a proper Date column from Year + Months
df['Date'] = pd.to_datetime(df['datetime'].astype(str) + '-' + df['Month_x'].astype(str) + '-01')

# Aggregate by Date (sum across LGAs for each month)
df_monthly = df.groupby('Date')[['Persons_fever','RDT_tested','positive_by RDT']].sum().reset_index()

# Plot malaria indicators over time
plt.figure(figsize=(12,6))
for col in ['Persons_fever','RDT_tested','positive_by RDT']:
    plt.plot(df_monthly['Date'], df_monthly[col], label=col, marker='o')

plt.legend()
plt.title("Monthly Malaria Indicators Over Time (Aggregated)")
plt.xlabel("Date")
plt.ylabel("Counts")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#insights:
#### There is a steady increase in the number of persons with fever, RDT-tested and positive by RDT from 2021 to 2023.
#### Additionally, from 2023 - 2024, there is a high spike of number of persosn with fever, tested by RDT and positive by RDT 
### And this spike could be due the flooding experience in Benue state in 2023 as well as the reduce use of intervention plans as deployed in 2020


# In[20]:


# plotting climate factors over time 
# Create a proper Date column from Year + Months
df['Date'] = pd.to_datetime(df['datetime'])


# Aggregate by Date (sum across LGAs for each month)
df_monthly = df.groupby('Date')[['Temperature (celsuis)']].mean().reset_index()

# Plot Average temperature over time
plt.figure(figsize=(12,6))
for col in ['Temperature (celsuis)']:
    plt.plot(df_monthly['Date'], df_monthly[col], label=col, marker='o')

plt.legend()
plt.title("Monthly climate Indicators Over Time (Aggregated)")
plt.xlabel("Date")
plt.ylabel("Counts")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[21]:


# plotting climate factors over time 
# Create a proper Date column from Year + Months
df['Date'] = pd.to_datetime(df['datetime'])

# Aggregate by Date (sum across LGAs for each month)
df_monthly = df.groupby('Date')[['Rainfall (mm)']].mean().reset_index()

# Plot Average Raifall over time
plt.figure(figsize=(12,6))
for col in ['Rainfall (mm)']:
    plt.plot(df_monthly['Date'], df_monthly[col], label=col, marker='o')

plt.legend()
plt.title("Monthly climate Indicators Over Time (Aggregated)")
plt.xlabel("Date")
plt.ylabel("Counts")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[22]:


# Correlation Analysis 
# Identify whether malaria prevalence is related to climate variables using the 



#selecting relevant variables 
malaria_vars = ["positive_by RDT", "positive_by_Microscopy", "Persons_fever"]
climate_vars = ["Rainfall (mm)", "Temperature (celsuis)"]

# subset dataframe to only include these
data = df[malaria_vars + climate_vars]

# spearman correlation

spearman_corr = data.corr(method = "spearman")

# === Pearson Correlation ===
pearson_corr = data.corr(method="pearson")

print("=== Pearson Correlation ===")
print(pearson_corr)
print("\n=== Spearman Correlation ===")
print(spearman_corr)
#Insights

## There is a weak relationship between the malaria key variables and climate factors such as rainfall and temperature as seen from the visuals
## Fever ↔ RDT positives is the strongest relationship (makes sense clinically).

## Microscopy doesn’t correlate as strongly with fever or RDT → may reflect differences in sensitivity/specificity of the tests.

## Climate factors (rainfall, temperature) show little same-month correlation with malaria outcomes, but this does not mean there’s no link — you likely need to check lagged correlations (e.g., rainfall in month t vs malaria in month t+1 or t+2).

## Rainfall and temperature have a clear inverse relationship, consistent with environmental dynamics.


# In[23]:


# Rename to match what pandas expects
df = df.rename(columns={"Year_y": "year", "Month_y": "month"})

# Ensuring the date column is properly assigned.
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1), errors='coerce')

# Sort by Date
df = df.sort_values('Date')

# Number of months to check lags for
max_lag = 6 

lagged_corrs = {}

for lag in range(1, max_lag+1):
    # Shift rainfall and temperature by "lag" months
    df[f'Rainfall_lag{lag}'] = df['Rainfall (mm)'].shift(lag)
    df[f'Temp_lag{lag}'] = df['Temperature (celsuis)'].shift(lag)
    
    # Calculate correlations with malaria outcome (RDT positives as example)
    pearson_rain = df[['positive_by RDT', f'Rainfall_lag{lag}']].corr(method='pearson').iloc[0,1]
    pearson_temp = df[['positive_by RDT', f'Temp_lag{lag}']].corr(method='pearson').iloc[0,1]
    
    spearman_rain = df[['positive_by RDT', f'Rainfall_lag{lag}']].corr(method='spearman').iloc[0,1]
    spearman_temp = df[['positive_by RDT', f'Temp_lag{lag}']].corr(method='spearman').iloc[0,1]
    
    # Save results
    lagged_corrs[lag] = {
        "Pearson Rainfall": pearson_rain,
        "Pearson Temperature": pearson_temp,
        "Spearman Rainfall": spearman_rain,
        "Spearman Temperature": spearman_temp
    }

# Convert results to a nice table
lagged_corrs_df = pd.DataFrame(lagged_corrs).T
print("=== Lagged Correlations (Rainfall & Temp vs RDT positives) ===")
print(lagged_corrs_df)

#insights 

####  from both correlation analysis Rainfall has a delayed effetct - malaria cases (positive_RDT) tends to rise a few months after increased rainfall, consistence with mosquitos breeding and transmission cycles
#### While temperature has a small but more immediate influence on malaria transmission compared to rainfall.
###### It is safe to say Rainfall effect lagged(delayed impact, clearer after 4-6 months) giving that mosquito populations expand with rainfall but malaria cases surge only months later once vector density and transimission clcles peak.
##### Indicating that malaria prevalence in Benue state infleuneced by climate variability, but not in a simple linear way
##### 


# In[23]:


## Visualizing the lagged correlation variables

from statsmodels.tsa.stattools import ccf
df_capy = df.copy()

# Drop missing values (important for time series)
data = df_capy.dropna(subset=['Rainfall (mm)','Temperature (celsuis)','positive_by RDT'])

# Extract the series
rainfall = data['Rainfall (mm)']
temperature = data['Temperature (celsuis)']
rdt = data['positive_by RDT']

# Compute CCF (rainfall vs RDT)
ccf_rainfall = ccf(rainfall, rdt)[:7]   # first 12 lags
ccf_temp = ccf(temperature, rdt)[:7]

lags = range(len(ccf_rainfall))

# Plot CCF for rainfall
plt.figure(figsize=(12,5))
plt.stem(lags, ccf_rainfall, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Cross-Correlation: Rainfall vs RDT positives")
plt.xlabel("Lag (months)")
plt.ylabel("CCF")
plt.show()

# Plot CCF for temperature
plt.figure(figsize=(12,5))
plt.stem(lags, ccf_temp, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Cross-Correlation: Temperature vs RDT positives")
plt.xlabel("Lag (months)")
plt.ylabel("CCF")
plt.show()

print(ccf_temp)


# In[23]:


## To understand the trend of malaria cases and temperature
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")


# Extract month names
df["Month"] = df["datetime"].dt.strftime("%b")  

# Group by month
malaria_by_month = df.groupby("Month")["positive_by RDT"].sum()
rainfall_avg_by_month = df.groupby("Month")["Temperature (celsuis)"].mean()

# Ensure correct month order
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
malaria_by_month = malaria_by_month.reindex(month_order)
rainfall_avg_by_month = rainfall_avg_by_month.reindex(month_order)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10,6))

# Malaria cases (bar chart)
ax1.bar(malaria_by_month.index, malaria_by_month.values, color="skyblue", label="Malaria Cases")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Malaria Cases", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Rainfall (average, line chart)
ax2 = ax1.twinx()
ax2.plot(rainfall_avg_by_month.index, rainfall_avg_by_month.values, color="red", marker="o", linewidth=2, label="Average Temperature (celsuis)")
ax2.set_ylabel("Average Temperature (celsuis)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Title + Legends
fig.suptitle("Malaria Cases vs Average Temperature", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

### As total cases spiked in March, with an average temperature of 32 degree celsuis and while this happened whcih indicate most seen situation of malaria prevalence relationship with temperature.
### Additionally as temperature decrease over the months there is an increase in the number of cases steadilty indicatin that temperature between 27-32 celsuis contribute to mosquitoes breeding and leading to increase in malaria cases as seen


# In[24]:


####==== creating visuals to understand the trends of climate parameters and positive by Rd

#  datetime column is parsed correctly
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Extract month names
df["Month"] = df["datetime"].dt.strftime("%b")  # e.g., Jan, Feb, Mar

# Group by month
malaria_by_month = df.groupby("Month")["positive_by RDT"].sum()
rainfall_avg_by_month = df.groupby("Month")["Rainfall (mm)"].mean()

# Ensure correct month order
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
malaria_by_month = malaria_by_month.reindex(month_order)
rainfall_avg_by_month = rainfall_avg_by_month.reindex(month_order)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10,6))

# Malaria cases (bar chart)
ax1.bar(malaria_by_month.index, malaria_by_month.values, color="skyblue", label="Malaria Cases")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Malaria Cases", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Rainfall (average, line chart)
ax2 = ax1.twinx()
ax2.plot(rainfall_avg_by_month.index, rainfall_avg_by_month.values, color="red", marker="o", linewidth=2, label="Average Rainfall (mm)")
ax2.set_ylabel("Average Rainfall (mm)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Title + Legends
fig.suptitle("Malaria Cases vs Average Rainfall", fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
# the rain contribute to an increase in the number of cases as seen from the visual, cases spiked as the rainfall increases


# In[25]:


# Make a temporary copy with datetime as index
df_log = df.copy()
df_log = df_log.set_index("datetime").sort_index()


max_lag = 6  

for lag in range(1, max_lag + 1):
    df_log[f"Rainfall_lag{lag}"] = df_log["Rainfall (mm)"].shift(lag)
    df_log[f"Temperature_lag{lag}"] = df_log["Temperature (celsuis)"].shift(lag)

lags = range(1, 7)
corrs = {
    "Rainfall": [df_log["positive_by RDT"].corr(df_log[f"Rainfall_lag{lag}"]) for lag in lags],
    "Temperature": [df_log["positive_by RDT"].corr(df_log[f"Temperature_lag{lag}"]) for lag in lags]
}

# Compute correlations 
rain_corrs = [df_log["positive_by RDT"].corr(df_log[f"Rainfall_lag{lag}"]) for lag in lags]
temp_corrs = [df_log["positive_by RDT"].corr(df_log[f"Temperature_lag{lag}"]) for lag in lags]

x = np.arange(len(lags))  # positions

width = 0.35  # bar width

plt.figure(figsize=(10,6))
plt.bar(x - width/2, rain_corrs, width, label="Rainfall")
plt.bar(x + width/2, temp_corrs, width, label="Temperature")

plt.xticks(x, [f"Lag {lag}" for lag in lags])
plt.ylabel("Correlation with Malaria Positivity (RDT)")
plt.title("Lagged Correlations: Rainfall vs Temperature")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Print correlation values for inspection
for lag in lags:
    print(f"Lag {lag}: "
          f"Rainfall corr = {rain_corrs[lag-1]:.3f}, "
          f"Temperature corr = {temp_corrs[lag-1]:.3f}")

### insights
#==== Rainfall
##Correlations are very weak (0.001–0.023).
##This suggests that rainfall in the previous 1–6 months has little to no linear relationship with malaria positivity in your dataset.
##Possible reasons:
##The effect of rainfall may be non-linear (e.g., too much rain washes away breeding sites).
##The relationship may depend on local geography or lagged interactions with temperature

#==== Temperature
# Correlations are consistently higher than rainfall (0.053–0.065).
# Although still weak (close to zero), the positive trend suggests that warmer temperatures 1–6 months earlier may slightly increase malaria positivity.
# This makes sense biologically: warmer climates favor mosquito survival and parasite development.

# Final thoughts: Temperature seems to be a more useful predictor of malaria positivity than rainfall in this dataset.Rainfall on its own is not strongly predictive — but rainfall combined with temperature


# # Case 1(without the lagged features) ML

# In[24]:


# In this case, the monthly rainfall and temperature are directly used to predict the positive by RDT
# which will then serve as the baseline for the lagged model.

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings



# 
# ### preparing X and Y

# In[50]:


X = df.drop(
    columns=[
        'Year_x', 
        'LGA', 
        'Microscopy_tested', 
        'positive_by_Microscopy', 
        'positive_by RDT',
        'datetime',
        'Month_x', 
        'Month_y', 
        'Year_y', 
        'Date'
    ], 
    axis=1
)


# In[51]:


X.tail()


# In[52]:


Y= df['positive_by RDT']


# In[53]:


Y


# In[54]:


# separate dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# ### 
# Create an Evaluate Function to give all metrics after model Training

# In[55]:


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[56]:


from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# Create a wrapper so it behaves like your sklearn models
class NegativeBinomialRegressor:
    def __init__(self):
        self.model = None
    
    def fit(self, X, Y):
        X = sm.add_constant(X)  # statsmodels needs intercept
        self.model = sm.GLM(Y, X, family=NegativeBinomial()).fit()
        return self
    
    def predict(self, X):
        X = sm.add_constant(X, has_constant='add')
        return self.model.predict(X)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Negative Binomial": NegativeBinomialRegressor()
}
model_list = []
r2_list =[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, Y_train) # Train model

    # Make predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Y_train, Y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Y_test, Y_test_pred)

    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')


##The number of malaria-positive cases by RDT can be reliably predicted using persons with fever, rainfall levels, and malaria case counts. A simple linear model already explains most of the relationship, suggesting that these predictors have a stable, direct effect on RDT-confirmed malaria cases. More complex models (like Random Forest) didn’t improve much, meaning the relationship is not overly nonlinear.


# ### Results

# In[57]:


pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)


# In[58]:


####  Linear Regression

lin_model = LinearRegression(fit_intercept=True)
lin_model = lin_model.fit(X_train, Y_train)
Y_pred = lin_model.predict(X_test)
score = r2_score(Y_test, Y_pred)*100
print(" Accuracy of the model is %.2f" %score)


# In[59]:


plt.scatter(Y_test,Y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[60]:


sns.regplot(x=Y_test,y=Y_pred,ci=None,color ='red');


# In[61]:


# Train Random Forest again separately
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, Y_train)

# Get feature importances
importances = rf_model.feature_importances_
features = X.columns  

# Put into a pandas Series for easy visualization
feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)

# Plot feature importance
plt.figure(figsize=(8,6))
feat_importances.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("Feature Importance - Random Forest Regressor", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.show()


# # ML with lag features 

# In[26]:


V = df_log.drop(
    columns=[
        'Year_x', 
        'Microscopy_tested', 
        'positive_by_Microscopy', 
        'positive_by RDT',
        'Month_x', 
        'Month_y', 
        'Year_y', 
        'Date'
    ], 
    axis=1
)


# In[27]:


V.tail()


# In[28]:


Z= df_log['positive_by RDT']


# In[29]:


Z


# In[30]:


# separate dataset into train and test
from sklearn.model_selection import train_test_split
V_train, V_test, Z_train, Z_test = train_test_split(V,Z,test_size=0.2,random_state=42)
V_train.shape, V_test.shape


# In[31]:


# Create Column Transformer with 3 types of transformers
num_features = V.select_dtypes(exclude="object").columns
cat_features = V.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
         ("StandardScaler", numeric_transformer, num_features),        
    ]
)


# In[32]:


V = preprocessor.fit_transform(V)


# In[31]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# ----------------------------
# Define evaluation function
# ----------------------------
def evaluate_model(Z_true, Z_pred):
    mae = mean_absolute_error(Z_true, Z_pred)
    rmse = np.sqrt(mean_squared_error(Z_true, Z_pred))
    r2 = r2_score(Z_true, Z_pred)
    return mae, rmse, r2

# ----------------------------
# Custom Negative Binomial Wrapper
# ----------------------------
class NegativeBinomialRegressor:
    def __init__(self):
        self.model = None
    
    def fit(self, V, Z):
        V = sm.add_constant(V)  # statsmodels needs intercept
        self.model = sm.GLM(Z, V, family=NegativeBinomial()).fit()
        return self
    
    def predict(self, V):
        V = sm.add_constant(V, has_constant='add')
        return self.model.predict(V)

# ----------------------------
# Prepare Data
# ----------------------------
# Assume your dataframe is df and:
# Z = target (positive RDT maybe)
# V = features (fever, rainfall, etc.)

V = df_log.drop(
    columns=[
        'Year_x',  
        'Microscopy_tested', 
        'positive_by_Microscopy', 
        'positive_by RDT',
        'Month_x', 
        'Month_y', 
        'Year_y', 
        'Date',
        'RDT_tested',
        'Persons_fever'
        
    ], 
    axis=1
)
Z = df_log["positive_by RDT"]  # target

# One-hot encode LGA
V = pd.get_dummies(V, columns=['LGA'], drop_first=True)

# Drop NaNs jointly so alignment is kept
data = pd.concat([V, Z], axis=1).dropna()
V = data.drop(columns=["positive_by RDT"])
Z = data["positive_by RDT"]


# Split train/test together
V_train, V_test, Z_train, Z_test = train_test_split(V, Z, test_size=0.2, random_state=42)

# ----------------------------
# Models
# ----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Negative Binomial": NegativeBinomialRegressor()
}

# ----------------------------
# Train + Evaluate
# ----------------------------
model_list = []
r2_list = []

for name, model in models.items():
    model.fit(V_train, Z_train)  # Train model

    # Predictions
    Z_train_pred = model.predict(V_train)
    Z_test_pred = model.predict(V_test)

    # Evaluate
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Z_train, Z_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Z_test, Z_test_pred)

    # Print results
    print(name)
    model_list.append(name)
    
    print('Model performance for Training set')
    print("- RMSE: {:.4f}".format(model_train_rmse))
    print("- MAE: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- RMSE: {:.4f}".format(model_test_rmse))
    print("- MAE: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*40)
    print('\n')





# In[32]:


import matplotlib.pyplot as plt
import numpy as np

# === Linear Regression Coefficients ===
lin_model = LinearRegression()
lin_model.fit(V_train, Z_train)

coefficients = lin_model.coef_
features = V_train.columns

plt.figure(figsize=(10,6))
plt.bar(features, coefficients)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Coefficient Value")
plt.title("Linear Regression Coefficients (Lagged Features)")
plt.show()

# === Random Forest Feature Importance ===
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(V_train, Z_train)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # sort descending

plt.figure(figsize=(10,6))
plt.bar([features[i] for i in indices], importances[indices])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance (Lagged Features)")
plt.show()


# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Correlation of features with malaria cases ---
corr_values = []
for col in V_train.columns:
    corr = np.corrcoef(V_train[col], Z_train)[0,1]
    corr_values.append(corr)

# --- Random Forest Importances ---
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(V_train, Z_train)
importances = rf_model.feature_importances_

# --- Build DataFrame for comparison ---
comparison_df = pd.DataFrame({
    "Feature": V_train.columns,
    "Correlation_with_Cases": corr_values,
    "RandomForest_Importance": importances
})

# --- Normalize values for heatmap (optional but clearer) ---
comparison_norm = comparison_df.copy()
comparison_norm["Correlation_with_Cases"] = comparison_norm["Correlation_with_Cases"] / comparison_norm["Correlation_with_Cases"].abs().max()
comparison_norm["RandomForest_Importance"] = comparison_norm["RandomForest_Importance"] / comparison_norm["RandomForest_Importance"].max()

# --- Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(comparison_norm.set_index("Feature").T, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Feature Correlation vs Random Forest Importance")
plt.ylabel("Metric")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')
plt.show()

# Display raw values for clarity
print(comparison_df.sort_values("RandomForest_Importance", ascending=False))


#Rainfall (current and lag1) and Temperature (especially lag6) are the top drivers of malaria cases.

# Correlation analysis alone is misleading (rainfall correlation looks weak), but RF highlights its predictive power — showing nonlinear or threshold-based effects.

# Temperature effects are delayed — strongest influence comes from 4–6 months earlier.

# Rainfall influence is more immediate (1 month lag) and cumulative (current rainfall).

# Mid-range rainfall lags (2–5 months) contribute little → possible redundancy or weak biological relevance.


# ### This suggests malaria dynamics are rainfall-driven in the short term and temperature-driven in the long term.
# 
# #### Without lags → models (like Linear/Ridge) had good accuracy but less biological realism.
# 
# #### With lags → Random Forest showed strong performance (R² ≈ 0.86 on test set) and feature importance clearly aligns with malaria biology.
# 
# 
# 
# 

# # Next Steps: Deployment
# 
# #### Deploy the lagged feature model on Streamlit.
# 
# #### Reason: It reflects real malaria transmission patterns (short-term rainfall + long-term temperature).
# 
# #### But  to keep it interpretable, we can:
# 
# #### Display top features (Rainfall current, Rainfall_lag1, Temperature_lag6).
# 
# #### Add short explanations: “Rainfall today and one month ago are key drivers. Temperature 6 months ago also plays a role.”
# 
# #### we can also let users toggle between “current features only” vs “lagged features” predictions, so they see the difference.

# # Streamlit ML Lagged feature Deployemnet (Random Forest)
# 
# 

# In[36]:


import joblib

for name, model in models.items():
    model.fit(V_train, Z_train)
    
    # Make predictions (optional for Random Forest saving)
    Z_train_pred = model.predict(V_train)
    Z_test_pred = model.predict(V_test)

    # Evaluate (same as before)
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Z_train, Z_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Z_test, Z_test_pred)

    print(name)
    print('Model performance for Test set')
    print("- RMSE: {:.4f}".format(model_test_rmse))
    print("- MAE: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print('='*40)

    # ---- SAVE ONLY RANDOM FOREST MODEL ----
    if name == "Random Forest Regressor":
        joblib.dump(model, "random_forest_model.pkl")
        print("Random Forest model saved as random_forest_model.pkl")


# In[37]:


rf_model = joblib.load("random_forest_model.pkl")
rf_predictions = rf_model.predict(V_test)


# In[41]:


import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

st.title("📊 Malaria Case Prediction App")

st.write("This app predicts malaria cases based on lagged environmental factors such as Rainfall(mm) and Temperature (celsuis). You can filter by LGA.")

# --- Input Widgets ---
# Example: LGAs dropdown
lgas = ["LGA_Ado", "LGA_Agatu", "LGA_Apa" , "LGA_Buruku" , "LGA_Gboko", "LGA_Guma", "LGA_Gwer East", "LGA_Gwer West", "LGA_Katsina-Ala", "LGA_Konshisha", "LGA_Kwanda", "LGA_Logo", "LGA_Makurdi", "LGA_Obi", "LGA_Ogbadibo", "LGA_Ohimini", "LGA_Oju", "LGA_Okpokwu", "LGA_Otukpo", "LGA_Tarka", "LGA_Ukum", "LGA_Ushongo", "LGA_Vandeikya"]  
selected_lga = st.selectbox("Select LGA", lgas)

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
Malaria_cases = st.number_input("Number of Malaria cases", min_value=0, step=1)

# --- Prepare Input Data ---
input_data = pd.DataFrame({
    "Rainfall": [rainfall],
    "Temperature": [temperature],
    "Persons_fever": [Malaria_cases],
    "LGA": [selected_lga]
})

# Encode categorical LGA (use same encoding as training)
input_data = pd.get_dummies(input_data, columns=["LGA"])

# Align columns with training data
# Save the feature column names used in training
joblib.dump(V.columns.tolist(), "feature_columns.pkl")

# You should save the training feature columns during training:
#   joblib.dump(V_train.columns, "feature_columns.pkl")
feature_columns = joblib.load("feature_columns.pkl")
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted malaria cases: {int(prediction)}")


# In[43]:


import joblib

# After training:
model = rf_model  # or whatever you called your trained model

# Save the trained model
joblib.dump(model, "random_forest_model.pkl")

# Save the feature columns used during training
joblib.dump(V.columns.tolist(), "feature_columns.pkl")

print("Model and feature columns saved!")


# In[ ]:




