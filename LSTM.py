import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# need to load database
from google.colab import drive
drive.mount('/content/drive')
loan_data = pd.read_csv('/content/drive/My Drive/Other/atlanta_financial.csv')
weather_data = pd.read_csv('/content/drive/My Drive/Other/atlanta_hartsfield_jackson_weather.csv')
employment_data = pd.read_csv('/content/drive/My Drive/Other/Employment__Unemployment__and_Labor_Force_Data.csv')
# TODO - add cpi after this works

# find weather data specifically for 2007 - 2019
weather_index_one = weather_data[weather_data.eq("2007-01-01").any(axis=1)].index[0]
weather_index_two = weather_data[weather_data.eq("2019-08-01").any(axis=1)].index[0]
weather_segment = weather_data['PRCP'][weather_index_one:(weather_index_two)].to_frame()

# find interest data specifically for 2007 - 2019
#initial_interest = loan_data["Original_Interest_Rate"].to_frame()
lower_bound = 2007
upper_bound = 2019
initial_interest = (loan_data[(loan_data['PayYear'] >= lower_bound) & (loan_data['PayYear'] <= upper_bound)]["Original_Interest_Rate"]).to_frame()

# given: employment data specifically for 2007 - 2019
employment_segment = employment_data["Employment Rate"].to_frame()

# replace later: target variable as Borrower_Credit_Score instead of default rate
credit_score = (loan_data[(loan_data['PayYear'] >= lower_bound) & (loan_data['PayYear'] <= upper_bound)]["Borrower_Credit_Score"].head(33)).to_frame()
mean_credit_score = credit_score['Borrower_Credit_Score'].mean()
credit_score['Borrower_Credit_Score'].fillna(mean_credit_score, inplace=True)

# step 1: build scaler

scaler = MinMaxScaler()
scaled_weather_data = scaler.fit_transform(weather_segment)
scaled_interest_data = scaler.fit_transform(initial_interest)
scaled_employment_segment = scaler.fit_transform(employment_segment)
scaled_credit_score = scaler.fit_transform(credit_score)

print(scaled_credit_score)

# step 2: concatenate to be one dataframe for input

concatenated_df = pd.concat([pd.DataFrame(scaled_weather_data), pd.DataFrame(scaled_interest_data), pd.DataFrame(scaled_employment_segment)], axis=1)

# cleaning dataframe of NaN values
concatenated_df = concatenated_df.dropna()
concatenated_df.columns = ['Weather', 'Interest', 'Employment']
concatenated_df = concatenated_df[(concatenated_df['Weather'] != 0) & (concatenated_df['Interest'] != 0) & (concatenated_df['Employment'] != 0)]

print(concatenated_df)
print(len(concatenated_df))

# step 3: define inputs and outputs

X = concatenated_df.values  # Input features
#y = credit_score.values     # Target variable

# Use MinMaxScaler to scale credit scores between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
credit_score_scaled = scaler.fit_transform(credit_score.values.reshape(-1, 1))

# Update the y variable
y = credit_score_scaled.reshape(-1, 1, 1)  # Adjust dimensions as per your model's requirement


#X = np.expand_dims(X, axis=1)  # This reshapes X to (num_samples, 1, num_features)

#print(X)

X = X.reshape((X.shape[0], 1, X.shape[1]))
y = y.reshape(y.shape[0], 1, y.shape[1])
print(X.shape)
print(y.shape)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=200, activation='relu', input_shape=(1, 3)))  # Use 3 because you have 3 features
model.add(Dense(1))

# input_shape=(sequence_length, input_features)
# Divide up data for training + testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

# Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
#model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

#model.compile(optimizer='adam', loss='mse')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.3)

# Assuming X_test is your test features and model is your trained LSTM
predictions = model.predict(X_test)
#predictions_y = model.predict(y_test)
print(predictions)
#print(predictions_y)

print("y_test shape:", y_test.shape)
print("Predictions shape:", predictions.shape)

# If your predictions array is something like (n_samples, 1, 1), reduce it to (n_samples,)
if predictions.ndim == 3 and predictions.shape[2] == 1:
    predictions = predictions.squeeze(axis=-1)

# Similarly, ensure y_test is also correctly shaped
if y_test.ndim == 3 and y_test.shape[2] == 1:
    y_test = y_test.squeeze(axis=-1)

# It's also common for y_test to be (n_samples, 1) when using data from Pandas or similar
if y_test.ndim == 2 and y_test.shape[1] == 1:
    y_test = y_test.squeeze(axis=-1)

# Calculate MSE, MAE, and RMSE
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)  # RMSE is just the square root of MSE

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

"""```
MSE: 0.1619
MAE: 0.3524
RMSE: 0.4023
```
"""

# Layers model - version 2

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

model = Sequential()
model.add(LSTM(500, return_sequences=True, input_shape=(1, 3), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(LSTM(500, return_sequences=False, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Calculate MSE, MAE, and RMSE
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)  # RMSE is just the square root of MSE

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

"""```
MSE: 0.1619
MAE: 0.3524
RMSE: 0.4023
```

# Data Augmentation

This is the Generative AI part of the project - augmenting the current dataset such that we can synthetically generate data, run the model, and make predictions with higher accuracy and lower MSE.
"""

def add_noise(data, noise_level=0.01): # noise function - feature function 1
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

def scale_features(data, scale_factor=0.1): # scaling function - feature function 2
    factors = 1 + scale_factor * (2 * np.random.rand(*data.shape) - 1)
    return data * factors

min_length = min(len(weather_segment), len(initial_interest), len(employment_segment)) # minimum length - ensure same database length
# redefine for the augmentation
weather_segment = weather_segment.iloc[:min_length]
initial_interest = initial_interest.iloc[:min_length]
employment_segment = employment_segment.iloc[:min_length]

#print(len(weather_segment))
#print(len(initial_interest))
#print(len(employment_segment))

# scaling + augmenting
scaled_weather_data = scaler.fit_transform(weather_segment)
scaled_interest_data = scaler.fit_transform(initial_interest)
weather_data_noisy = add_noise(scaled_weather_data, noise_level=0.02)
interest_data_noisy = add_noise(scaled_interest_data, noise_level=0.02)

print(len(scaled_weather_data))
print(len(scaled_interest_data))
print(len(weather_data_noisy))
print(len(interest_data_noisy))

augmented_df = pd.concat([
    pd.DataFrame(weather_data_noisy, columns=['Weather']),
    pd.DataFrame(interest_data_noisy, columns=['Interest']),
    pd.DataFrame(scaled_employment_segment[:min_length], columns=['Employment'])  # Make sure this matches in length
], axis=1)

# Proceed with model data preparation
augmented_df.dropna(inplace=True)
X_augmented = augmented_df.values
X_augmented = X_augmented.reshape((X_augmented.shape[0], 1, X_augmented.shape[1]))

# data check for same length
X_augmented = X_augmented[:33]
print(f"Length of X_augmented: {len(X_augmented)}")
print(f"Length of y: {len(y)}")

# Apply noise and scaling to the weather data
scaled_weather_data = scale_features(scaled_weather_data, scale_factor=0.05)
weather_data_noisy = add_noise(scaled_weather_data, noise_level=0.02)

# Apply noise to the interest data
interest_data_noisy = add_noise(scaled_interest_data, noise_level=0.01)

# Combine these into your dataframe
augmented_df = pd.concat([
    pd.DataFrame(weather_data_noisy, columns=['Weather']),
    pd.DataFrame(interest_data_noisy, columns=['Interest']),
    pd.DataFrame(scaled_employment_segment, columns=['Employment'])  # Assuming no augmentation here
], axis=1)

# Cleaning dataframe of NaN values might be needed again here
augmented_df.dropna(inplace=True)

# Debug: Check your augmented data
print(augmented_df.head())

print(f"Length of X_augmented: {len(X_augmented)}")
print(f"Length of y: {len(y)}")

# Define inputs and outputs using augmented data
X_augmented = augmented_df.values[:33]
X_augmented = X_augmented.reshape((X_augmented.shape[0], 1, X_augmented.shape[1]))

# Continue with the same y since target variables should not be artificially generated in this context
y = y.reshape(y.shape[0], 1, y.shape[1])

# Split the augmented dataset
#print(len(X_augmented))
#print(len(y))
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_augmented, y, test_size=0.3, random_state=42)

# Train your model on the augmented dataset
model.fit(X_train_aug, y_train_aug, epochs=50, batch_size=32, validation_split=0.3)

# Evaluate model with augmented data
predictions_aug = model.predict(X_test_aug)
mse_aug = mean_squared_error(y_test_aug.squeeze(), predictions_aug.squeeze())
print(f"Augmented MSE: {mse_aug}")

"""Augmented MSE: 0.10037484131757542"""

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.hist(initial_interest, bins=30)
plt.title('Original Interest Rates')

plt.subplot(1, 3, 2)
plt.hist(scaled_interest_data, bins=30)
plt.title('Scaled Interest Rates')

plt.subplot(1, 3, 3)
plt.hist(interest_data_noisy, bins=30)
plt.title('Noise Added Interest Rates')
plt.show()

"""# Testing Accuracy and Performance"""

#print(X_test)
#print(y_test)
model.compile(optimizer='adam', loss='mean_squared_error')

loss_original = model.evaluate(X_test, y_test)

"""According to the model, we have a final loss of 0.1021. Here, the loss represents the MSE (mean squared error), which has a relatively and respectably low value of 0.1021.

Now, let us see what happens when we calculate the same values for the augmented data.
"""

loss_aug = model.evaluate(X_test_aug, y_test_aug)

"""It seems that the augmented data has a loss of 0.1024, which is a fraction higher than the loss on the original test data. This would indicate that the above augmentations should be adjusted further (noise, scale) to further improve the model."""

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
results = model.evaluate(X_test, y_test)
print(f"Loss (MSE): {results[0]}")  # Assuming loss is MSE
print(f"Mean Absolute Error: {results[1]}")

"""The mean absolute error is 0.269176.

So, let us adjust the noise functions and run the models again.
"""

def add_noise(data, noise_level=0.005): # noise function - feature function 1 v2
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

def scale_features(data, scale_factor=0.05): # scaling function - feature function 2 v2
    factors = 1 + scale_factor * (2 * np.random.rand(*data.shape) - 1)
    return data * factors

min_length = min(len(weather_segment), len(initial_interest), len(employment_segment)) # minimum length - ensure same database length
# redefine for the augmentation
weather_segment = weather_segment.iloc[:min_length]
initial_interest = initial_interest.iloc[:min_length]
employment_segment = employment_segment.iloc[:min_length]

#print(len(weather_segment))
#print(len(initial_interest))
#print(len(employment_segment))

# scaling + augmenting
scaled_weather_data = scaler.fit_transform(weather_segment)
scaled_interest_data = scaler.fit_transform(initial_interest)
weather_data_noisy = add_noise(scaled_weather_data, noise_level=0.02)
interest_data_noisy = add_noise(scaled_interest_data, noise_level=0.02)

print(len(scaled_weather_data))
print(len(scaled_interest_data))
print(len(weather_data_noisy))
print(len(interest_data_noisy))

augmented_df = pd.concat([
    pd.DataFrame(weather_data_noisy, columns=['Weather']),
    pd.DataFrame(interest_data_noisy, columns=['Interest']),
    pd.DataFrame(scaled_employment_segment[:min_length], columns=['Employment'])  # Make sure this matches in length
], axis=1)

# Proceed with model data preparation
augmented_df.dropna(inplace=True)
X_augmented = augmented_df.values
X_augmented = X_augmented.reshape((X_augmented.shape[0], 1, X_augmented.shape[1]))

# data check for same length
X_augmented = X_augmented[:33]
print(f"Length of X_augmented: {len(X_augmented)}")
print(f"Length of y: {len(y)}")

# Apply noise and scaling to the weather data
scaled_weather_data = scale_features(scaled_weather_data, scale_factor=0.05)
weather_data_noisy = add_noise(scaled_weather_data, noise_level=0.02)

# Apply noise to the interest data
interest_data_noisy = add_noise(scaled_interest_data, noise_level=0.01)

# Combine these into your dataframe
augmented_df = pd.concat([
    pd.DataFrame(weather_data_noisy, columns=['Weather']),
    pd.DataFrame(interest_data_noisy, columns=['Interest']),
    pd.DataFrame(scaled_employment_segment, columns=['Employment'])  # Assuming no augmentation here
], axis=1)

# Cleaning dataframe of NaN values might be needed again here
augmented_df.dropna(inplace=True)

# Debug: Check your augmented data
print(augmented_df.head())

print(f"Length of X_augmented: {len(X_augmented)}")
print(f"Length of y: {len(y)}")

# Define inputs and outputs using augmented data
X_augmented = augmented_df.values[:33]
X_augmented = X_augmented.reshape((X_augmented.shape[0], 1, X_augmented.shape[1]))

# Continue with the same y since target variables should not be artificially generated in this context
y = y.reshape(y.shape[0], 1, y.shape[1])

# Split the augmented dataset
#print(len(X_augmented))
#print(len(y))
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_augmented, y, test_size=0.3, random_state=42)

# Train your model on the augmented dataset
model.fit(X_train_aug, y_train_aug, epochs=50, batch_size=32, validation_split=0.3)

# Evaluate model with augmented data
predictions_aug = model.predict(X_test_aug)
mse_aug = mean_squared_error(y_test_aug.squeeze(), predictions_aug.squeeze())
print(f"Augmented MSE: {mse_aug}")

"""The most optimal value of Augmented MSE that could be determined using the noise and scale functions was 0.10009260298674341. This was at noise value of 0.005 and a scale value of 0.05. Now, we will do documentation and data visualization."""

# Example: Adjusting noise level
noise_level = 0.005  # Reduced from 0.01 if original was too disruptive
X_train_noisy = add_noise(X_train_aug, noise_level=noise_level)

# Re-train the model with adjusted data
model.fit(X_train_noisy, y_train, epochs=50, batch_size=32, validation_split=0.3)

predictions_aug = model.predict(X_test_aug)
mse_aug = mean_squared_error(y_test_aug.squeeze(), predictions_aug.squeeze())
print(f"Augmented MSE: {mse_aug}")

results_log = {
    'experiment1': {'noise_level': 0.05, 'loss': 0.10037484131757542},
    'experiment2': {'noise_level': 0.005, 'loss': 0.10008957652108656}
}
print(results_log)

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True Values')
plt.plot(predictions, label='Predictions')
plt.title('Comparison of True Values and Model Predictions')
plt.legend()
plt.show()