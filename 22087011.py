# Import Libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

def read_and_clean_data(file_path):
    """
    Read data from a CSV file, replace '..' with NaN, convert
    columns to numeric, replace NaN with mean,
    and return both the original data and the transposed cleaned data.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - orig_data (pd.DataFrame): The original data.
    - cleaned_data (pd.DataFrame): The cleaned data.
    - cleaned_data_transposed (pd.DataFrame): The transposed cleaned data.
    """
    # Read the data from the file
    df = pd.read_csv('dataset.csv')
    orig_data = df.copy()

    # Replace '..' with NaN
    df = df.replace('..', np.nan)

    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace NaN with mean
    df = df.fillna(df.mean())

    # Transpose the cleaned data
    cleaned_data_transposed = df.transpose()

    return orig_data, df, cleaned_data_transposed


def linear_model(x, a, b):
    """
    Linear model function for curve fitting.

    Parameters:
    - x (array-like): Independent variable.
    - a (float): Slope of the line.
    - b (float): Intercept of the line.

    Returns:
    - array-like: Fitted values.
    """
    return a * x + b


def confidence_interval(x, params, covariance, conf=0.95):
    """
    Calculate the confidence interval for the curve fit parameters.

    Parameters:
    - x (array-like): Independent variable.
    - params (array-like): Parameters of the model.
    - covariance (array-like): Covariance matrix of the parameters.
    - conf (float, optional): Confidence level. Default is 0.95.

    Returns:
    - tuple: Lower and upper bounds of the confidence interval.
    """
    perr = np.sqrt(np.diag(covariance))
    alpha = 1 - conf
    nstd = stats.norm.ppf(1 - alpha / 2)
    lower = linear_model(x, *(params - nstd * perr))
    upper = linear_model(x, *(params + nstd * perr))
    return lower, upper


# Read and clean the data
orig_data, cleaned_data, cleaned_data_transposed = read_and_clean_data('p3.csv')
print('-------------')
print(orig_data['Country Name'])

# Select relevant columns for clustering
clustering_columns = ['Adjusted net national income (annual % growth) [NY.ADJ.NNTY.KD.ZG]',
                       'Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]',
                       'Adjusted net savings, excluding particulate emission damage (% of GNI) [NY.ADJ.SVNX.GN.ZS]',
                       'Adjusted net savings, including particulate emission damage (% of GNI) [NY.ADJ.SVNG.GN.ZS]',
                       'Gross savings (% of GDP) [NY.GNS.ICTR.ZS]',
                       'CO2 emissions from gaseous fuel consumption (kt) [EN.ATM.CO2E.GF.KT]',
                       'CO2 emissions (metric tons per capita) [EN.ATM.CO2E.PC]',
                       'CO2 emissions from gaseous fuel consumption (% of total) [EN.ATM.CO2E.GF.ZS]']

data_for_clustering = cleaned_data[clustering_columns]

# Normalize the data
normalized_data = (data_for_clustering - data_for_clustering.mean()) / data_for_clustering.std()

# Silhouette Score Calculation
kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)
silhouette_avg = silhouette_score(normalized_data, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Assuming you have normalized data and applied K-Means to 'cleaned_data'
kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)

# Plotting the clustered data
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='Adjusted net national income (annual % growth) [NY.ADJ.NNTY.KD.ZG]',
                          y='Gross savings (% of GDP) [NY.GNS.ICTR.ZS]',
                          hue='Cluster',
                          data=cleaned_data,
                          palette='viridis',
                          legend='full')

# Plotting cluster centers
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red',
            marker='X', s=200, label='Cluster Centers')

plt.title('Clustering of Countries')
plt.xlabel('Adjusted net national income (annual % growth)')
plt.ylabel('Gross savings (% of GDP)')
plt.legend()
plt.show()

# Curve fitting for a specific country (e.g., Hungary)
curve_data = orig_data[orig_data['Country Name'] == 'Hungary']
time_data = curve_data['Time']
gross_savings_data = curve_data['Gross savings (% of GDP) [NY.GNS.ICTR.ZS]']

# Replace '..' with NaN in the gross_savings_data
gross_savings_data = gross_savings_data.replace('..', np.nan)

# Convert to numeric and drop NaN values
time_data_numeric, gross_savings_data_numeric = pd.to_numeric(time_data, errors='coerce'), pd.to_numeric(
    gross_savings_data, errors='coerce')
valid_data_mask = ~np.isnan(time_data_numeric) & ~np.isnan(gross_savings_data_numeric)

# Curve fitting using cleaned data
params, covariance = curve_fit(linear_model, time_data_numeric[valid_data_mask],
                               gross_savings_data_numeric[valid_data_mask])

# Calculate confidence interval
lower_bound, upper_bound = confidence_interval(time_data_numeric, params, covariance)

# Plotting the data, fitted curve, and confidence interval
plt.figure(figsize=(10, 6))
plt.scatter(time_data_numeric, gross_savings_data_numeric, label='Original Data')
plt.plot(time_data_numeric, linear_model(time_data_numeric, *params), label='Fitted Curve', color='red')
plt.fill_between(time_data_numeric, lower_bound, upper_bound, color='lightgreen', alpha=0.3,
                 label='Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Gross savings (% of GDP)')
plt.ylim(0, 50)  # Set y-axis range to [0, 50]
plt.legend()
plt.title('Curve Fitting with Confidence Interval for Time and Gross Savings')
plt.show()

# Years for prediction
years_to_predict = [2024, 2029]

# Predicted values for Gross savings (% of GDP) for each country
predicted_values = {}

for country in ['Hungary']:
    # Extract time data for the specific country
    time_data_country = curve_data[curve_data['Country Name'] == country]['Time']

    # Predict Gross savings (% of GDP) for the specified years
    predicted_values[country] = [linear_model(year, *params) for year in years_to_predict]

# Display the predicted values
for country, values in predicted_values.items():
    print(f'{country}:')
    for year, predicted_value in zip(years_to_predict, values):
        print(f'  Year {year}: {predicted_value:.2f}%')
    print()

# Perform curve fitting using cleaned data for the extended time range
params, covariance = curve_fit(linear_model, time_data_numeric[valid_data_mask],
                               gross_savings_data_numeric[valid_data_mask])

# Extend the time range for plotting
extended_time_range = np.arange(1995, 2030, 1)

# Generate fitted values for the extended time range
fit_gross_savings_extended = linear_model(extended_time_range, *params)

# Generate predicted values for the extended time range
predicted_values_extended = [linear_model(year, *params) for year in extended_time_range]

# Plotting the data, fitted curve, and predicted values with y-axis range [15, 23]
plt.figure(figsize=(10, 6))
plt.scatter(time_data_numeric, gross_savings_data_numeric, label='Original Data')
plt.plot(extended_time_range, fit_gross_savings_extended, label='Fitted Curve', color='red')
plt.scatter(extended_time_range, predicted_values_extended, label='Predicted Values', color='green', marker='x')
plt.fill_between(time_data_numeric, lower_bound, upper_bound, color='lightgreen', alpha=0.3,
                 label='Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Gross savings (% of GDP)')
plt.ylim(1, 40)  # Set y-axis range to [15, 23]
plt.legend()
plt.title('Curve Fitting with Confidence Interval and Predicted Values for Time and Gross Savings in Hungary')
plt.show()
