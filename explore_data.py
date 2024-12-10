import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# LEARNING SET
# ======================================================================================================================


directory = "given documents/LS/"
# Load subject IDs
subject_data = np.loadtxt(f"{directory}subject_Id.txt")
# Load activity data
activity_data = np.loadtxt(f"{directory}activity_Id.txt")

# Initialize a list to hold missing value indicators for each sensor
missing_matrix = []

# Loop through sensor files and check for missing values
for sensor_id in range(2, 33):  # Assuming sensor files are named LS_sensor_2.txt to LS_sensor_32.txt
    file_name = f"{directory}LS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    # Check if the entire row (time series) is missing
    missing_indicator = np.any(sensor_data == -999999.99, axis=1)
    missing_matrix.append(missing_indicator)

# Convert the list to a DataFrame for easier visualization
missing_matrix = np.array(missing_matrix).T  # Transpose to have rows as samples and columns as sensors

# Create a DataFrame to pair subject IDs with missing information
missing_df = pd.DataFrame(missing_matrix, columns=[f"Sensor_{i}" for i in range(2, 33)])
missing_df['Subject_ID'] = subject_data


# Calculate the percentage of entrances with at least one missing value
entrances_with_missing_values = np.any(missing_matrix, axis=1).sum()
total_entrances = missing_matrix.shape[0]
percentage_missing = (entrances_with_missing_values / total_entrances) * 100

print(f"Number of entrances with at least one missing value: {entrances_with_missing_values}")
print(f"Percentage of entrances with at least one missing value: {percentage_missing:.2f}%")


# ABSOLUTE VALUES

# Visualization 1: Heatmap of missing data across all sensors
plt.figure(figsize=(10, 8))
sns.heatmap(missing_matrix.T, cbar=False, cmap="coolwarm")
plt.xlabel("Samples (Time Series)")
plt.ylabel("Sensors")
plt.title("Missing Data Heatmap Across Sensors")
plt.show()

# Visualization 2: Missing time series count per sensor
missing_counts = missing_matrix.sum(axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(2, 33), missing_counts)
plt.xlabel("Sensor ID")
plt.ylabel("Count of Missing Time Series")
plt.title("Missing Values per Sensor")
plt.xticks(range(2, 33))
plt.show()

# Visualization 3: Missing time series grouped by subject
missing_by_subject = missing_df.groupby('Subject_ID').apply(lambda x: x.iloc[:, :-1].any(axis=1).sum())
plt.figure(figsize=(10, 6))
plt.bar(missing_by_subject.index, missing_by_subject.values)
plt.xlabel("Subject ID")
plt.ylabel("Count of Missing Time Series")
plt.title("Missing Data per Subject")
plt.xticks(sorted(missing_by_subject.index.astype(int)))
plt.show()

# RELATIVE MISSING VALUES

# Calculate relative missing values per sensor
total_time_series = missing_matrix.shape[0]  # Total number of samples (time series)
relative_missing_per_sensor = (missing_counts / total_time_series) * 100# Load activity data

# Plot relative missing values per sensor
plt.figure(figsize=(10, 6))
plt.bar(range(2, 33), relative_missing_per_sensor)
plt.xlabel("Sensor ID")
plt.ylabel("Percentage of Missing Time Series")
plt.title("Relative Missing Values per Sensor")
plt.xticks(range(2, 33))
plt.show()

# Calculate relative missing values per subject
total_time_series_per_subject = missing_df.groupby('Subject_ID').size()
missing_by_subject_relative = (missing_by_subject / total_time_series_per_subject) * 100

# Plot relative missing values per subject
plt.figure(figsize=(10, 6))
plt.bar(missing_by_subject_relative.index, missing_by_subject_relative.values)
plt.xlabel("Subject ID")
plt.ylabel("Percentage of Missing Time Series")
plt.title("Relative Missing Data per Subject")
plt.xticks(sorted(missing_by_subject_relative.index.astype(int)))
plt.show()

# Calculate relative missing values per activity
total_time_series_per_activity = pd.Series(activity_data).value_counts()  # Total time series per activity
missing_by_activity = pd.DataFrame(missing_matrix).groupby(activity_data).apply(lambda x: x.any(axis=1).sum())
missing_by_activity_relative = (missing_by_activity / total_time_series_per_activity) * 100

# Plot relative missing values per activity
plt.figure(figsize=(10, 6))
plt.bar(missing_by_activity_relative.index, missing_by_activity_relative.values)
plt.xlabel("Activity ID")
plt.ylabel("Percentage of Missing Time Series")
plt.title("Relative Missing Data per Activity")
plt.xticks(sorted(missing_by_activity_relative.index.astype(int)))
plt.show()


# DISTRIBUTION OF SUBJECTS AND ACTIVITIES

# Plot distribution of subjects
plt.figure(figsize=(10, 6))
subject_counts = pd.Series(subject_data).value_counts()
plt.bar(subject_counts.index, subject_counts.values)
plt.xlabel("Subject ID")
plt.ylabel("Count of Time Series")
plt.title("Distribution of Time Series by Subject")
plt.xticks(sorted(subject_counts.index.astype(int)))
plt.show()

# Plot distribution of activities
plt.figure(figsize=(10, 6))
activity_counts = pd.Series(activity_data).value_counts()
plt.bar(activity_counts.index, activity_counts.values)
plt.xlabel("Activity ID")
plt.ylabel("Count of Time Series")
plt.title("Distribution of Time Series by Activity")
plt.xticks(sorted(activity_counts.index.astype(int)))
plt.show()


# MEAN AND STANDARD DEVIATION OF SENSOR DATA

# Calculate statistics for each sensor

print(f"{'Sensor':<10}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")

for sensor_id in range(2, 33):
    file_name = f"{directory}LS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    valid_data = sensor_data[sensor_data != -999999.99]  # Exclude missing values
    print(f"{sensor_id:<10}{valid_data.mean():>10.2f}{valid_data.std():>10.2f}{valid_data.min():>10.2f}{valid_data.max():>10.2f}")


# ======================================================================================================================









# TEST SET
# ======================================================================================================================

directory = "given documents/TS/"
# Load subject IDs
subject_data = np.loadtxt(f"{directory}subject_Id.txt")

# Initialize a list to hold missing value indicators for each sensor
missing_matrix = []

# Loop through sensor files and check for missing values
for sensor_id in range(2, 33):  # Assuming sensor files are named TS_sensor_2.txt to TS_sensor_32.txt
    file_name = f"{directory}TS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    # Check if the entire row (time series) is missing
    missing_indicator = np.any(sensor_data == -999999.99, axis=1)
    missing_matrix.append(missing_indicator)

# Convert the list to a DataFrame for easier visualization
missing_matrix = np.array(missing_matrix).T  # Transpose to have rows as samples and columns as sensors

# Create a DataFrame to pair subject IDs with missing information
missing_df = pd.DataFrame(missing_matrix, columns=[f"Sensor_{i}" for i in range(2, 33)])
missing_df['Subject_ID'] = subject_data

# ABSOLUTE VALUES

# Visualization 1: Heatmap of missing data across all sensors
plt.figure(figsize=(10, 8))
sns.heatmap(missing_matrix.T, cbar=False, cmap="coolwarm")
plt.xlabel("Samples (Time Series)")
plt.ylabel("Sensors")
plt.title("Missing Data Heatmap Across Sensors (Test Set)")
plt.show()

# Visualization 2: Missing time series count per sensor
missing_counts = missing_matrix.sum(axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(2, 33), missing_counts)
plt.xlabel("Sensor ID")
plt.ylabel("Count of Missing Time Series")
plt.title("Missing Values per Sensor (Test Set)")
plt.xticks(range(2, 33))
plt.show()

# Visualization 3: Missing time series grouped by subject
missing_by_subject = missing_df.groupby('Subject_ID').apply(lambda x: x.iloc[:, :-1].any(axis=1).sum())
plt.figure(figsize=(10, 6))
plt.bar(missing_by_subject.index, missing_by_subject.values)
plt.xlabel("Subject ID")
plt.ylabel("Count of Missing Time Series")
plt.title("Missing Data per Subject (Test Set)")
plt.xticks(sorted(missing_by_subject.index.astype(int)))
plt.show()

# RELATIVE MISSING VALUES

# Calculate relative missing values per sensor
total_time_series = missing_matrix.shape[0]  # Total number of samples (time series)
relative_missing_per_sensor = (missing_counts / total_time_series) * 100

# Plot relative missing values per sensor
plt.figure(figsize=(10, 6))
plt.bar(range(2, 33), relative_missing_per_sensor)
plt.xlabel("Sensor ID")
plt.ylabel("Percentage of Missing Time Series")
plt.title("Relative Missing Values per Sensor (Test Set)")
plt.xticks(range(2, 33))
plt.show()

# Calculate relative missing values per subject
total_time_series_per_subject = missing_df.groupby('Subject_ID').size()
missing_by_subject_relative = (missing_by_subject / total_time_series_per_subject) * 100

# Plot relative missing values per subject
plt.figure(figsize=(10, 6))
plt.bar(missing_by_subject_relative.index, missing_by_subject_relative.values)
plt.xlabel("Subject ID")
plt.ylabel("Percentage of Missing Time Series")
plt.title("Relative Missing Data per Subject (Test Set)")
plt.xticks(sorted(missing_by_subject_relative.index.astype(int)))
plt.show()

# DISTRIBUTION OF SUBJECTS

# Plot distribution of subjects
plt.figure(figsize=(10, 6))
subject_counts = pd.Series(subject_data).value_counts()
plt.bar(subject_counts.index, subject_counts.values)
plt.xlabel("Subject ID")
plt.ylabel("Count of Time Series")
plt.title("Distribution of Time Series by Subject (Test Set)")
plt.xticks(sorted(subject_counts.index.astype(int)))
plt.show()

# MEAN AND STANDARD DEVIATION OF SENSOR DATA

# Calculate statistics for each sensor

print(f"{'Sensor':<10}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")

for sensor_id in range(2, 33):
    file_name = f"{directory}TS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    valid_data = sensor_data[sensor_data != -999999.99]  # Exclude missing values
    print(f"{sensor_id:<10}{valid_data.mean():>10.2f}{valid_data.std():>10.2f}{valid_data.min():>10.2f}{valid_data.max():>10.2f}")


# ======================================================================================================================


# COMBINED SET
# ======================================================================================================================

# Combine the learning and test sets for sensor data
combined_missing_matrix = []

for sensor_id in range(2, 33):
    ls_file_name = f"given documents/LS/LS_sensor_{sensor_id}.txt"
    ts_file_name = f"given documents/TS/TS_sensor_{sensor_id}.txt"
    ls_sensor_data = np.loadtxt(ls_file_name)
    ts_sensor_data = np.loadtxt(ts_file_name)
    combined_sensor_data = np.concatenate((ls_sensor_data, ts_sensor_data))
    combined_missing_matrix.append(combined_sensor_data)

# Calculate statistics for each sensor in the combined set
print(f"{'Sensor':<10}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")

for sensor_id, sensor_data in zip(range(2, 33), combined_missing_matrix):
    valid_data = sensor_data[sensor_data != -999999.99]  # Exclude missing values
    print(f"{sensor_id:<10}{valid_data.mean():>10.2f}{valid_data.std():>10.2f}{valid_data.min():>10.2f}{valid_data.max():>10.2f}")


# ======================================================================================================================


# Compare statistics of the learning set with those of the test set

# Initialize dictionaries to hold statistics for learning and test sets
learning_stats = {}
test_stats = {}

# Calculate statistics for learning set
for sensor_id in range(2, 33):
    file_name = f"given documents/LS/LS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    valid_data = sensor_data[sensor_data != -999999.99]  # Exclude missing values
    learning_stats[sensor_id] = {
        'mean': valid_data.mean(),
        'std': valid_data.std(),
        'min': valid_data.min(),
        'max': valid_data.max()
    }

# Calculate statistics for test set
for sensor_id in range(2, 33):
    file_name = f"given documents/TS/TS_sensor_{sensor_id}.txt"
    sensor_data = np.loadtxt(file_name)
    valid_data = sensor_data[sensor_data != -999999.99]  # Exclude missing values
    test_stats[sensor_id] = {
        'mean': valid_data.mean(),
        'std': valid_data.std(),
        'min': valid_data.min(),
        'max': valid_data.max()
    }

# Print comparison of statistics
print(f"{'Sensor':<10}{'LS Mean':>10}{'TS Mean':>10}{'LS Std':>10}{'TS Std':>10}{'LS Min':>10}{'TS Min':>10}{'LS Max':>10}{'TS Max':>10}")

for sensor_id in range(2, 33):
    ls_stats = learning_stats[sensor_id]
    ts_stats = test_stats[sensor_id]
    print(f"{sensor_id:<10}{ls_stats['mean']:>10.2f}{ts_stats['mean']:>10.2f}{ls_stats['std']:>10.2f}{ts_stats['std']:>10.2f}{ls_stats['min']:>10.2f}{ts_stats['min']:>10.2f}{ls_stats['max']:>10.2f}{ts_stats['max']:>10.2f}")