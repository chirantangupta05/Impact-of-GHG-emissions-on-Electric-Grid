# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:33:13 2022

@author: Chirantan.Gupta
"""

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Matplotlib visualization
import matplotlib.pyplot as plt


# Set default font size
plt.rcParams['font.size'] = 24

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



####Read the data ###################################################
data = pd.read_csv(r'C:\Users\Chirantan.Gupta\OneDrive - Shell\Desktop\Decarbonization_Hackathon\NYC_Property_Energy.csv')




# Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available': np.nan})

data.dtypes

for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Convert the data type to float
        data[col] = data[col].astype(float)
        
# Statistics for each column
data.describe()

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
missing_vals_data = missing_values_table(data)

# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

# Drop the columns
data = data.drop(columns = list(missing_columns))

figsize(8, 8)

# Rename the score 
data = data.rename(columns = {'ENERGY STAR Score': 'score'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 
plt.title('Energy Star Score Distribution');


# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');

data['Site EUI (kBtu/ft²)'].describe()

data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10)

# Calculate first and third quartile
first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
            (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');

####Delete the column order which is redundant#####################
data.drop(columns =['Order'], axis =1 , inplace = True)

#######Relationships between attributes
###Find out all numeric attributes

df_numerics_only = data.select_dtypes(include=np.number)


###Find out all categorical attributes
df_categorical_only = data.select_dtypes(include=['object'])

import matplotlib.patches as  mpatches
from dython.nominal import associations

###Drop all nan in both categoricals_only and numericals_only dataframes

df_categorical_only_na_removed = df_categorical_only.dropna()

categorical_correlation= associations(df_categorical_only_na_removed,cramers_v_bias_correction=False, filename= 'categorical_correlation.png', figsize=(10,10))

df_categorical_corr=categorical_correlation['corr']
df_categorical_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)


df_numerics_only_na_removed = df_numerics_only.dropna()

numerics_correlation= associations(df_numerics_only_na_removed,cramers_v_bias_correction=False, filename= 'numerics_correlation.png', figsize=(10,10))

df_numerics_corr=numerics_correlation['corr']
df_numerics_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)


data_drop_na = data.dropna()
complete_correlation= associations(data_drop_na,cramers_v_bias_correction=False, filename= 'All_correlation.png', figsize=(50,50))

df_complete_corr=complete_correlation['corr']
df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)




# Create a list of buildings with more than 83 measurements
types = data.dropna(subset=['Site EUI (kBtu/ft²)'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)

# Plot of distribution of scores for building categories
figsize(24, 20)

# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['Largest Property Use Type'] == b_type]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['Site EUI (kBtu/ft²)'].dropna(),
               label = b_type, shade = False, alpha = 0.8);
    
    
# label the plot
plt.legend()
plt.xlabel('Site EUI (kBtu/ft²)', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Site EUI (kBtu/ft²) by Building Type', size = 28);
###Correlation between Site EUI (kBtu/ft²) and Largest Property Use Type is 35%
###Correlation between Site EUI (kBtu/ft²) and Street Number is 70.2%
###Correlation between Site EUI (kBtu/ft²) and Borough is approx. 7%

####To be included in categorical variables: 
    ###Street Number, Street Name(?),List of All Property Use Types at Property,Largest Property Use Type ,Borough
    ###List of All Property Use Types at Property,Largest Property Use Type have a correlation of approximately 95%
    ###Street Name and Borough have very high correlation= 97%
    ###Street Number is more sensible to use as it has approx. 65% correlation with Street Name
    ###So Largest Property Use Type and Borough can be selected as they have 20% correlation
    ###Also Street Number can be used as it has 1% correlation with Borough
    ###Also Street Number and Largest Property Use Type have a 27.5% correlation
    ###Street Number can also be selected but it is useless

df_numerics_only_copy = df_numerics_only.copy()    

# Create columns with square root and log of numeric columns
for col in df_numerics_only_copy.columns:
    # Skip the Energy Star Score column
    if col == 'Site EUI (kBtu/ft²)':
        next
    else:
        df_numerics_only_copy['sqrt_' + col] = np.sqrt(df_numerics_only_copy[col])
        df_numerics_only_copy['log_' + col] = np.log(df_numerics_only_copy[col])

# Select the categorical columns
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([df_numerics_only_copy, categorical_subset], axis = 1)

# Drop buildings without a 'Site EUI (kBtu/ft²)'
features = features.dropna(subset = ['Site EUI (kBtu/ft²)'])

# Find correlations with the 'Site EUI (kBtu/ft²)'
correlations = features.corr()['Site EUI (kBtu/ft²)'].dropna().sort_values()

# Display most positive correlations
correlations.tail(15)

figsize(12, 10)

# Extract the building types
features['Largest Property Use Type'] = data.dropna(subset = ['Site EUI (kBtu/ft²)'])['Largest Property Use Type']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['Largest Property Use Type'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('score', 'Site EUI (kBtu/ft²)',
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Energy Star Score", size = 28)
plt.ylabel('Site EUI (kBtu/ft²)', size = 28)
plt.title('Site EUI vs Energy Star Score', size = 36);

# Extract the columns to  plot
plot_data = features[['score', 'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Source EUI (kBtu/ft²)', 
                      'log_Total GHG Emissions (Metric Tons CO2e)']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns 
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
                                        'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})

# Drop na values
plot_data = plot_data.dropna()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3)

# Upper is a scatter plot
grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# Bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

# Title for entire plot
plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02);

# Copy the original data
features = data.copy()

# Select the numeric columns
numeric_subset = data.select_dtypes('number')
"""
# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'Site EUI (kBtu/ft²)':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
"""        
# Select the categorical columns
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

features.shape

plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()
#plot_data.corr()
plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
plt.xlabel('Site EUI')
plt.ylabel('Weather Norm EUI')
plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % 0.996783);

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between Site EUI (kBtu/ft²)
    y = x[['Site EUI (kBtu/ft²)','Electricity Use - Grid Purchase (kBtu)','Total GHG Emissions (Metric Tons CO2e)']]
    x = x.drop(columns = ['Site EUI (kBtu/ft²)','Electricity Use - Grid Purchase (kBtu)','Total GHG Emissions (Metric Tons CO2e)'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 
                          'Water Use (All Water Sources) (kgal)',
                          
                          'Largest Property Use Type - Gross Floor Area (ft²)'])
    
    # Add the Site EUI (kBtu/ft²) back in to the data
    x[['Site EUI (kBtu/ft²)','Electricity Use - Grid Purchase (kBtu)','Total GHG Emissions (Metric Tons CO2e)']] = y
               
    return x


# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(features, 0.6);


# Remove any columns with all na values
features  = features.dropna(axis=1, how = 'all')
features.shape

# Extract the buildings with no Site EUI (kBtu/ft²) and the buildings with a score
no_Site_EUI= features[features['Site EUI (kBtu/ft²)'].isna()]
Site_EUI = features[features['Site EUI (kBtu/ft²)'].notnull()]

print(no_Site_EUI.shape)
print(Site_EUI.shape)

# Separate out the features and targets
features = Site_EUI.drop(columns='Site EUI (kBtu/ft²)')
targets = pd.DataFrame(Site_EUI['Site EUI (kBtu/ft²)'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 80% training and 20% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X.shape)##(9052, 65)
print(X_test.shape)##(2263, 65)
print(y.shape)##(9052, 1)
print(y_test.shape)##(2263, 1)

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


baseline_guess = np.median(y)

print('The baseline guess is a Site EUI of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))


train_features = X
test_features = X_test
train_labels = y
test_labels = y_test


# Display sizes of data
print('Training Feature Size: ', train_features.shape)
print('Testing Feature Size:  ', test_features.shape)
print('Training Labels Size:  ', train_labels.shape)
print('Testing Labels Size:   ', test_labels.shape)

figsize(8, 8)

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(train_labels['Site EUI (kBtu/ft²)'].dropna(), bins = 100);
plt.xlabel('Site EUI (kBtu/ft²)');
plt.ylabel('Number of Buildings'); 
plt.title('Site EUI Distribution');

###Finding out columns which have only one value
train_features.columns[train_features.nunique(dropna=True) <= 1]
##'Largest Property Use Type_Pre-school/Daycare'

test_features.columns[test_features.nunique(dropna=True) <= 1]

"""
'Largest Property Use Type_Bank Branch',
       'Largest Property Use Type_Convenience Store without Gas Station',
       'Largest Property Use Type_Courthouse',
       'Largest Property Use Type_Enclosed Mall',
       'Largest Property Use Type_Library',
       'Largest Property Use Type_Mailing Center/Post Office',
       'Largest Property Use Type_Other - Recreation',
       'Largest Property Use Type_Other - Services',
       'Largest Property Use Type_Performing Arts',
       'Largest Property Use Type_Refrigerated Warehouse',
       'Largest Property Use Type_Restaurant',
       'Largest Property Use Type_Social/Meeting Hall',
       'Largest Property Use Type_Urgent Care/Clinic/Other Outpatient',
       'Largest Property Use Type_Wholesale Club/Supercenter'
"""

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Train on the training features
imputer.fit(train_features)

# Transform both training data and testing data
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)  

print('Missing values in training features: ', np.sum(np.isnan(X)))##0
print('Missing values in testing features:  ', np.sum(np.isnan(X_test))) ##0

# Make sure all values are finite
print(np.where(~np.isfinite(X)))##No Record
print(np.where(~np.isfinite(X_test)))   ## No Record 

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))

from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK

from hyperopt.pyll import scope

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
import pickle
import sys
import copy
import statistics as stat

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import findspark
findspark.init()

mlflow.end_run()

from scipy.stats import loguniform
search_space = {
           
           'n_estimators':[100, 500, 900, 1100, 1500],
           'max_depth':[2, 3, 5, 10, 15,20],
           "learning_rate": loguniform(0.01, 1),##learning rate
           'subsample':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
           'min_samples_leaf':[1, 2, 4, 6, 8, 10],
           'min_samples_split':[2,4,6,8,10],
           'max_features': [0.1,0.3,0.5,0.7,0.8,1.0]
           }

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=search_space,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X, y)

# Get all of the cv results and sort by the test performance
random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

random_results.head(10)

random_cv.best_estimator_

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 10,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.7,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.08185951005222196,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = ['neg_mean_absolute_error','r2'], verbose = 1,
                           n_jobs = -1, return_train_score = True,refit='r2')

# Fit the grid search
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], results['mean_test_r2'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], results['mean_train_r2'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean r2'); plt.legend();
plt.title('Performance vs Number of Trees');

results.sort_values('mean_test_r2', ascending = False).head(5)

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model2 = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 10,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.7,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.08185951005222196,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search2 = GridSearchCV(estimator = model2, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)

# Fit the grid search
grid_search2.fit(X, y)

# Get the results into a dataframe
results2 = pd.DataFrame(grid_search2.cv_results_)

# Plot the training and testing error vs number of trees
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_test_score'], label = 'Testing Error')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');


# Select the best model
final_model = grid_search.best_estimator_

final_model

importance = grid_search.best_estimator_.feature_importances_

index_values = [i for i in range(0,65)]
column_values = ['Variable_Importance']

importance_df = pd.DataFrame(data= importance,index = index_values,columns = column_values )
importance_df["Variables"] = train_features.columns

top_10= importance_df.nlargest(10,'Variable_Importance')

"""
   Variable_Importance                                          Variables
64             0.122294             Total GHG Emissions (Metric Tons CO2e)
63             0.098616             Electricity Use - Grid Purchase (kBtu)
2              0.094412                               DOF Gross Floor Area
6              0.092844                                              score
7              0.086018  Weather Normalized Site Electricity Intensity ...
0              0.084614                                        Property Id
8              0.070721  Weather Normalized Site Natural Gas Intensity ...
9              0.066462      Water Intensity (All Water Sources) (gal/ft²)
11             0.063472                                          Longitude
3              0.060386                                         Year Built
"""
final_model.fit(X, y)

final_pred = final_model.predict(X_test)

print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))
##Final model performance on the test set:   MAE = 5.3212.
print('Final model performance on the test set:   R2_score = %0.4f.' % r2_score(y_test, final_pred))
###Final model performance on the test set:   R2_score = 0.9341.
figsize(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(final_pred, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot
plt.xlabel('Site EUI (kBtu/ft²)'); plt.ylabel('Density');plt.legend();
plt.title('Test Values and Predictions');

figsize = (6, 6)

# Calculate the residuals 
residuals = final_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); plt.ylabel('Count')
plt.title('Distribution of Residuals');

###############10% reduction of total ghg emissions results##############
####Final model performance on the test set:   MAE = 5.3212.
###Final model performance on the test set:   R2_score = 0.9341.

"""
    Variable_Importance                                          Variables
64             0.122294             Total GHG Emissions (Metric Tons CO2e)
63             0.098616             Electricity Use - Grid Purchase (kBtu)
2              0.094412                               DOF Gross Floor Area
6              0.092844                                              score
7              0.086018  Weather Normalized Site Electricity Intensity ...
0              0.084614                                        Property Id
8              0.070721  Weather Normalized Site Natural Gas Intensity ...
9              0.066462      Water Intensity (All Water Sources) (gal/ft²)
11             0.063472                                          Longitude
3              0.060386                                         Year Built

"""

#################30% reduction of total ghg emissions result

###Final model performance on the test set:   MAE = 5.3212.
###Final model performance on the test set:   R2_score = 0.9341.


"""
    Variable_Importance                                          Variables
64             0.122294             Total GHG Emissions (Metric Tons CO2e)
63             0.098616             Electricity Use - Grid Purchase (kBtu)
2              0.094412                               DOF Gross Floor Area
6              0.092844                                              score
7              0.086018  Weather Normalized Site Electricity Intensity ...
0              0.084614                                        Property Id
8              0.070721  Weather Normalized Site Natural Gas Intensity ...
9              0.066462      Water Intensity (All Water Sources) (gal/ft²)
11             0.063472                                          Longitude
3              0.060386                                         Year Built

"""
#####################50% reduction of total ghg emissions result

###Final model performance on the test set:   MAE = 5.3212.
###Final model performance on the test set:   R2_score = 0.9341.


"""
    Variable_Importance                                          Variables
64             0.122294             Total GHG Emissions (Metric Tons CO2e)
63             0.098616             Electricity Use - Grid Purchase (kBtu)
2              0.094412                               DOF Gross Floor Area
6              0.092844                                              score
7              0.086018  Weather Normalized Site Electricity Intensity ...
0              0.084614                                        Property Id
8              0.070721  Weather Normalized Site Natural Gas Intensity ...
9              0.066462      Water Intensity (All Water Sources) (gal/ft²)
11             0.063472                                          Longitude
3              0.060386                                         Year Built

"""
### 90% reduction of Total GHG emissions result
###Final model performance on the test set:   MAE = 5.3212.
###Final model performance on the test set:   R2_score = 0.9341.



"""
   Variable_Importance                                          Variables
64             0.122294             Total GHG Emissions (Metric Tons CO2e)
63             0.098616             Electricity Use - Grid Purchase (kBtu)
2              0.094412                               DOF Gross Floor Area
6              0.092844                                              score
7              0.086018  Weather Normalized Site Electricity Intensity ...
0              0.084614                                        Property Id
8              0.070721  Weather Normalized Site Natural Gas Intensity ...
9              0.066462      Water Intensity (All Water Sources) (gal/ft²)
11             0.063472                                          Longitude
3              0.060386                                         Year Built

"""
###########################################################################################
##############################Reducing GHG Emissions by 10% /50% /90% to see what are the impacts..

train_features2 = train_features.copy()
test_features2 = test_features.copy()
train_labels2 = train_labels.copy()
test_labels2 = test_labels.copy()



def change_percentage(df, column, percent):
    if percent < -100:
        raise ValueError("Percent out of range")
    multiplier = 1 - percent / 100
    df[column] *= multiplier
    
change_percentage(train_features2, 'Total GHG Emissions (Metric Tons CO2e)', 90)
change_percentage(test_features2, 'Total GHG Emissions (Metric Tons CO2e)', 90)

# Display sizes of data
print('Training Feature Size: ', train_features2.shape)
print('Testing Feature Size:  ', test_features2.shape)
print('Training Labels Size:  ', train_labels2.shape)
print('Testing Labels Size:   ', test_labels2.shape)


figsize= (8, 8)

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(train_labels2['Site EUI (kBtu/ft²)'].dropna(), bins = 100);
plt.xlabel('Site EUI (kBtu/ft²)');
plt.ylabel('Number of Buildings'); 
plt.title('Site EUI Distribution');

###Finding out columns which have only one value
train_features2.columns[train_features2.nunique(dropna=True) <= 1]
##'Largest Property Use Type_Pre-school/Daycare'

test_features2.columns[test_features2.nunique(dropna=True) <= 1]

"""
['Largest Property Use Type_Bank Branch',
       'Largest Property Use Type_Convenience Store without Gas Station',
       'Largest Property Use Type_Courthouse',
       'Largest Property Use Type_Enclosed Mall',
       'Largest Property Use Type_Library',
       'Largest Property Use Type_Mailing Center/Post Office',
       'Largest Property Use Type_Other - Recreation',
       'Largest Property Use Type_Other - Services',
       'Largest Property Use Type_Performing Arts',
       'Largest Property Use Type_Refrigerated Warehouse',
       'Largest Property Use Type_Restaurant',
       'Largest Property Use Type_Social/Meeting Hall',
       'Largest Property Use Type_Urgent Care/Clinic/Other Outpatient',
       'Largest Property Use Type_Wholesale Club/Supercenter']

"""
# Create an imputer object with a median filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Train on the training features
imputer.fit(train_features2)

# Transform both training data and testing data
X = imputer.transform(train_features2)
X_test = imputer.transform(test_features2)  

print('Missing values in training features: ', np.sum(np.isnan(X)))##0
print('Missing values in testing features:  ', np.sum(np.isnan(X_test))) ##0


# Make sure all values are finite
print(np.where(~np.isfinite(X)))##No Record
print(np.where(~np.isfinite(X_test)))   ## No Record 

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels2).reshape((-1, ))
y_test = np.array(test_labels2).reshape((-1, ))

from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK

from hyperopt.pyll import scope

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
import pickle
import sys
import copy
import statistics as stat

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
import findspark
findspark.init()

mlflow.end_run()

from scipy.stats import loguniform


search_space = {
           
           'n_estimators':[100, 500, 900, 1100, 1500],
           'max_depth':[2, 3, 5, 10, 15,20],
           "learning_rate": loguniform(0.01, 1),##learning rate
           'subsample':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
           'min_samples_leaf':[1, 2, 4, 6, 8, 10],
           'min_samples_split':[2,4,6,8,10],
           'max_features': [0.1,0.3,0.5,0.7,0.8,1.0]
           }

# Create the model to use for hyperparameter tuning
model3 = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model3,
                               param_distributions=search_space,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X, y)


# Get all of the cv results and sort by the test performance
random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

random_results.head(10)

random_cv.best_estimator_


"""
GradientBoostingRegressor(learning_rate=0.040596116104843046, max_depth=20,
                          max_features=0.8, min_samples_leaf=6,
                          random_state=42)
"""


# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}


          
model = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 10,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.7,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.08185951005222196,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = ['neg_mean_absolute_error','r2'], verbose = 1,
                           n_jobs = -1, return_train_score = True,refit='r2')

# Fit the grid search
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], results['mean_test_r2'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], results['mean_train_r2'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean r2'); plt.legend();
plt.title('Performance vs Number of Trees');

results.sort_values('mean_test_r2', ascending = False).head(5)

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model2 = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 10,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.7,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.08185951005222196,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search2 = GridSearchCV(estimator = model2, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)

# Fit the grid search
grid_search2.fit(X, y)

# Get the results into a dataframe
results2 = pd.DataFrame(grid_search2.cv_results_)

# Plot the training and testing error vs number of trees
figsize= (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_test_score'], label = 'Testing Error')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');

# Select the best model
final_model = grid_search.best_estimator_

final_model

importance = grid_search.best_estimator_.feature_importances_

index_values = [i for i in range(0,65)]
column_values = ['Variable_Importance']

importance_df = pd.DataFrame(data= importance,index = index_values,columns = column_values )
importance_df["Variables"] = train_features.columns

top_10= importance_df.nlargest(10,'Variable_Importance')

final_model.fit(X, y)

final_pred = final_model.predict(X_test)

print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))
print('Final model performance on the test set:   R2_score = %0.4f.' % r2_score(y_test, final_pred))

figsize=(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(final_pred, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot
plt.xlabel('Site EUI (kBtu/ft²)'); plt.ylabel('Density');plt.legend();
plt.title('Test Values and Predictions');

figsize = (6, 6)

# Calculate the residuals 
residuals = final_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); plt.ylabel('Count')
plt.title('Distribution of Residuals');



#######################################################################################################



####We will concentrate on 'Weather Normalized Site Electricity Intensity (kWh/ft²)' 
####And 'Weather Normalized Site Natural Gas Intensity (therms/ft²)' to see the performance
"""
Weather Normalized Site Electricity Intensity (kWh/ft²):
    
Weather Normalized Site Energy divided by property size
or by flow through a water/wastewater treatment plant.
"""

train_features3 = train_features.copy()
test_features3 = test_features.copy()
train_labels3 = train_labels.copy()
test_labels3 = test_labels.copy()

train_features3.drop(columns = ['Electricity Use - Grid Purchase (kBtu)'], axis =1 , inplace = True)
test_features3.drop(columns = ['Electricity Use - Grid Purchase (kBtu)'], axis =1 , inplace = True)
##'Total GHG Emissions (Metric Tons CO2e)'
train_features3.drop(columns = ['Total GHG Emissions (Metric Tons CO2e)'], axis =1 , inplace = True)
test_features3.drop(columns = ['Total GHG Emissions (Metric Tons CO2e)'], axis =1 , inplace = True)

def change_percentage(df, column, percent):
    if percent < -100:
        raise ValueError("Percent out of range")
    multiplier = 1 - percent / 100
    df[column] *= multiplier
    
change_percentage(train_features3, 'Weather Normalized Site Electricity Intensity (kWh/ft²)', 30)
change_percentage(test_features3, 'Weather Normalized Site Electricity Intensity (kWh/ft²)', 30)


# Display sizes of data
print('Training Feature Size: ', train_features3.shape)
print('Testing Feature Size:  ', test_features3.shape)
print('Training Labels Size:  ', train_labels3.shape)
print('Testing Labels Size:   ', test_labels3.shape)


figsize= (8, 8)

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(train_labels3['Site EUI (kBtu/ft²)'].dropna(), bins = 100);
plt.xlabel('Site EUI (kBtu/ft²)');
plt.ylabel('Number of Buildings'); 
plt.title('Site EUI Distribution');


###Finding out columns which have only one value
train_features3.columns[train_features3.nunique(dropna=True) <= 1]
##'Largest Property Use Type_Pre-school/Daycare'
test_features3.columns[test_features3.nunique(dropna=True) <= 1]
"""
['Largest Property Use Type_Bank Branch',
       'Largest Property Use Type_Convenience Store without Gas Station',
       'Largest Property Use Type_Courthouse',
       'Largest Property Use Type_Enclosed Mall',
       'Largest Property Use Type_Library',
       'Largest Property Use Type_Mailing Center/Post Office',
       'Largest Property Use Type_Other - Recreation',
       'Largest Property Use Type_Other - Services',
       'Largest Property Use Type_Performing Arts',
       'Largest Property Use Type_Refrigerated Warehouse',
       'Largest Property Use Type_Restaurant',
       'Largest Property Use Type_Social/Meeting Hall',
       'Largest Property Use Type_Urgent Care/Clinic/Other Outpatient',
       'Largest Property Use Type_Wholesale Club/Supercenter']

"""

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Train on the training features
imputer.fit(train_features3)

# Transform both training data and testing data
X = imputer.transform(train_features3)
X_test = imputer.transform(test_features3)  

print('Missing values in training features: ', np.sum(np.isnan(X)))##0
print('Missing values in testing features:  ', np.sum(np.isnan(X_test))) ##0


# Make sure all values are finite
print(np.where(~np.isfinite(X)))##No Record
print(np.where(~np.isfinite(X_test)))   ## No Record 

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels3).reshape((-1, ))
y_test = np.array(test_labels3).reshape((-1, ))



search_space = {
           
           'n_estimators':[100, 500, 900, 1100, 1500],
           'max_depth':[2, 3, 5, 10, 15,20],
           "learning_rate": loguniform(0.01, 1),##learning rate
           'subsample':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
           'min_samples_leaf':[1, 2, 4, 6, 8, 10],
           'min_samples_split':[2,4,6,8,10],
           'max_features': [0.1,0.3,0.5,0.7,0.8,1.0]
           }

# Create the model to use for hyperparameter tuning
model3 = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model3,
                               param_distributions=search_space,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X, y)

random_results.head(10)

random_cv.best_estimator_

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 20,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.8,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.040596116104843046,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = ['neg_mean_absolute_error','r2'], verbose = 1,
                           n_jobs = -1, return_train_score = True,refit='r2')

# Fit the grid search
grid_search.fit(X, y)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], results['mean_test_r2'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], results['mean_train_r2'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean r2'); plt.legend();
plt.title('Performance vs Number of Trees');

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model2 = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 20,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.8,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.040596116104843046,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search2 = GridSearchCV(estimator = model2, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)

# Fit the grid search
grid_search2.fit(X, y)

# Get the results into a dataframe
results2 = pd.DataFrame(grid_search2.cv_results_)

# Plot the training and testing error vs number of trees
figsize= (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_test_score'], label = 'Testing Error')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');

# Select the best model
final_model = grid_search.best_estimator_

final_model

importance = grid_search.best_estimator_.feature_importances_

index_values = [i for i in range(0,63)]
column_values = ['Variable_Importance']

importance_df = pd.DataFrame(data= importance,index = index_values,columns = column_values )
importance_df["Variables"] = train_features3.columns

top_10= importance_df.nlargest(10,'Variable_Importance')

"""
Final model performance on the test set:   MAE = 8.7307.
Final model performance on the test set:   R2_score = 0.8284.
    Variable_Importance                                          Variables
0              0.124042                                        Property Id
2              0.120926                               DOF Gross Floor Area
6              0.105071                                              score
7              0.104940  Weather Normalized Site Electricity Intensity ...
9              0.091506      Water Intensity (All Water Sources) (gal/ft²)
3              0.088495                                         Year Built
11             0.086927                                          Longitude
10             0.080785                                           Latitude
8              0.070171  Weather Normalized Site Natural Gas Intensity ...
13             0.066842                                       Census Tract
"""
final_model.fit(X, y)

final_pred = final_model.predict(X_test)

print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))
print('Final model performance on the test set:   R2_score = %0.4f.' % r2_score(y_test, final_pred))

figsize=(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(final_pred, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot
plt.xlabel('Site EUI (kBtu/ft²)'); plt.ylabel('Density');plt.legend();
plt.title('Test Values and Predictions');

figsize = (6, 6)

# Calculate the residuals 
residuals = final_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); plt.ylabel('Count')
plt.title('Distribution of Residuals');


#################################################################################################################


## Focussing on 'Weather Normalized Site Natural Gas Intensity (therms/ft²)' to see the performance
"""
Weather Normalized Site Natural Gas Intensity (therms/ft²)

Weather Normalized Site Energy divided by property size
or by flow through a water/wastewater treatment plant

"""
train_features4 = train_features.copy()
test_features4 = test_features.copy()
train_labels4 = train_labels.copy()
test_labels4 = test_labels.copy()

train_features4.drop(columns = ['Electricity Use - Grid Purchase (kBtu)'], axis =1 , inplace = True)
test_features4.drop(columns = ['Electricity Use - Grid Purchase (kBtu)'], axis =1 , inplace = True)
##'Total GHG Emissions (Metric Tons CO2e)'
train_features4.drop(columns = ['Total GHG Emissions (Metric Tons CO2e)'], axis =1 , inplace = True)
test_features4.drop(columns = ['Total GHG Emissions (Metric Tons CO2e)'], axis =1 , inplace = True)

def change_percentage(df, column, percent):
    if percent < -100:
        raise ValueError("Percent out of range")
    multiplier = 1 - percent / 100
    df[column] *= multiplier
    
change_percentage(train_features4, 'Weather Normalized Site Natural Gas Intensity (therms/ft²)', 30)
change_percentage(test_features4, 'Weather Normalized Site Natural Gas Intensity (therms/ft²)', 30)

# Display sizes of data
print('Training Feature Size: ', train_features4.shape)
print('Testing Feature Size:  ', test_features4.shape)
print('Training Labels Size:  ', train_labels4.shape)
print('Testing Labels Size:   ', test_labels4.shape)

figsize= (8, 8)

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(train_labels4['Site EUI (kBtu/ft²)'].dropna(), bins = 100);
plt.xlabel('Site EUI (kBtu/ft²)');
plt.ylabel('Number of Buildings'); 
plt.title('Site EUI Distribution');

###Finding out columns which have only one value
train_features4.columns[train_features4.nunique(dropna=True) <= 1]
##'Largest Property Use Type_Pre-school/Daycare'
test_features4.columns[test_features4.nunique(dropna=True) <= 1]
"""
['Largest Property Use Type_Bank Branch',
       'Largest Property Use Type_Convenience Store without Gas Station',
       'Largest Property Use Type_Courthouse',
       'Largest Property Use Type_Enclosed Mall',
       'Largest Property Use Type_Library',
       'Largest Property Use Type_Mailing Center/Post Office',
       'Largest Property Use Type_Other - Recreation',
       'Largest Property Use Type_Other - Services',
       'Largest Property Use Type_Performing Arts',
       'Largest Property Use Type_Refrigerated Warehouse',
       'Largest Property Use Type_Restaurant',
       'Largest Property Use Type_Social/Meeting Hall',
       'Largest Property Use Type_Urgent Care/Clinic/Other Outpatient',
       'Largest Property Use Type_Wholesale Club/Supercenter']
"""

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Train on the training features
imputer.fit(train_features4)

# Transform both training data and testing data
X = imputer.transform(train_features4)
X_test = imputer.transform(test_features4)  

print('Missing values in training features: ', np.sum(np.isnan(X)))##0
print('Missing values in testing features:  ', np.sum(np.isnan(X_test))) ##0


# Make sure all values are finite
print(np.where(~np.isfinite(X)))##No Record
print(np.where(~np.isfinite(X_test)))   ## No Record 

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels4).reshape((-1, ))
y_test = np.array(test_labels4).reshape((-1, ))

search_space = {
           
           'n_estimators':[100, 500, 900, 1100, 1500],
           'max_depth':[2, 3, 5, 10, 15,20],
           "learning_rate": loguniform(0.01, 1),##learning rate
           'subsample':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
           'min_samples_leaf':[1, 2, 4, 6, 8, 10],
           'min_samples_split':[2,4,6,8,10],
           'max_features': [0.1,0.3,0.5,0.7,0.8,1.0]
           }

# Create the model to use for hyperparameter tuning
model3 = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model3,
                               param_distributions=search_space,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X, y)

random_results.head(10)

random_cv.best_estimator_

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 20,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.8,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.040596116104843046,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = ['neg_mean_absolute_error','r2'], verbose = 1,
                           n_jobs = -1, return_train_score = True,refit='r2')

# Fit the grid search
grid_search.fit(X, y)

# Plot the training and testing error vs number of trees
figsize=(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], results['mean_test_r2'], label = 'Testing Error')
plt.plot(results['param_n_estimators'], results['mean_train_r2'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean r2'); plt.legend();
plt.title('Performance vs Number of Trees');

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}
           
model2 = GradientBoostingRegressor(loss = 'absolute_error', max_depth = 20,
                                  min_samples_leaf = 6,
                                  min_samples_split = 6,
                                  max_features = 0.8,
                                  n_estimators=500,
                                  subsample=0.6,
                                  learning_rate=0.040596116104843046,
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search2 = GridSearchCV(estimator = model2, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)

# Fit the grid search
grid_search2.fit(X, y)

# Get the results into a dataframe
results2 = pd.DataFrame(grid_search2.cv_results_)

# Plot the training and testing error vs number of trees
figsize= (8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_test_score'], label = 'Testing Error')
plt.plot(results2['param_n_estimators'], -1 * results2['mean_train_score'], label = 'Training Error')
plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
plt.title('Performance vs Number of Trees');

# Select the best model
final_model = grid_search.best_estimator_

final_model

importance = grid_search.best_estimator_.feature_importances_

index_values = [i for i in range(0,63)]
column_values = ['Variable_Importance']

importance_df = pd.DataFrame(data= importance,index = index_values,columns = column_values )
importance_df["Variables"] = train_features3.columns

top_10= importance_df.nlargest(10,'Variable_Importance')

"""
Final model performance on the test set:   MAE = 8.7307

Final model performance on the test set:   R2_score = 0.8284.

    Variable_Importance                                          Variables
0              0.124042                                        Property Id
2              0.120926                               DOF Gross Floor Area
6              0.105071                                              score
7              0.104940  Weather Normalized Site Electricity Intensity ...
9              0.091506      Water Intensity (All Water Sources) (gal/ft²)
3              0.088495                                         Year Built
11             0.086927                                          Longitude
10             0.080785                                           Latitude
8              0.070171  Weather Normalized Site Natural Gas Intensity ...
13             0.066842                                       Census Tract
"""

final_model.fit(X, y)

final_pred = final_model.predict(X_test)

print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))
print('Final model performance on the test set:   R2_score = %0.4f.' % r2_score(y_test, final_pred))

figsize=(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(final_pred, label = 'Predictions')
sns.kdeplot(y_test, label = 'Values')

# Label the plot
plt.xlabel('Site EUI (kBtu/ft²)'); plt.ylabel('Density');plt.legend();
plt.title('Test Values and Predictions');

figsize = (6, 6)

# Calculate the residuals 
residuals = final_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color = 'red', bins = 20,
         edgecolor = 'black')
plt.xlabel('Error'); plt.ylabel('Count')
plt.title('Distribution of Residuals');

