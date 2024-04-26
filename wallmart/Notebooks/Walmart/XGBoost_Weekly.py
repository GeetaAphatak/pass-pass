import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import datetime

base_folder = 'C:\\Geeta\\learning\\projects\\AnomalyDetectionSXM\\Notebooks\\Datasets\\Pipeline'
dataset_name = 'Walmart_Weekly'
train_file = base_folder + '/train/' + dataset_name + '_train.csv'
inference_file = base_folder + '/inference/' + dataset_name + '_inference.csv'

current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_file_path = base_folder + '/xgboostADmodel_' + dataset_name + current_date_time + '.pkl'

inference_results_file = base_folder + '/inference/' + dataset_name + '_inference_results.csv'

target = 'Weekly_Sales'
# target = 'Daily_Sales'

# Group the data by 'State' and perform lag shifting within each group
groupby_cols = ['State']

# Define lag columns to consider
lag_columns = ['Weekly_Sales', 'Temperature']  # , 'Fuel_Price', 'CPI', 'Unemployment']

# Define the lag values to be used
lags = [1, 2, 4]  # Example lag values of 1 week and 2 weeks

state_encoder = 'label_encoder_State.pkl'
data_path = train_file


def read(data_path, sheet_name='', usecols=None):
    df = pd.DataFrame()
    if data_path.split('.')[-1] == 'xlsx':
        if sheet_name:
            df = pd.read_excel(data_path, sheet_name=sheet_name, usecols=usecols)
        else:
            df = pd.read_excel(data_path, usecols=usecols)
        print("Shape of the data in file {} is {}".format(data_path, df.shape))
    else:
        try:
            df = pd.read_csv(data_path)
            print("Shape of the data in file {} is {}".format(data_path, df.shape))
            if df.shape[0] == 0:
                print("No data in file {}".format(data_path))
        except Exception as e:
            print("Issue while reading data at {} \n{}".format(data_path, e))
    return df


def standardize_date_col(dataframe, date_col):
    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format='%d-%m-%Y', errors='coerce').fillna(
        pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce'))
    # Convert all dates to 'mm-dd-yyyy' format
    dataframe[date_col] = dataframe[date_col].dt.strftime('%Y-%m-%d')
    return dataframe


# Read data from csv or excel, sheet_name is the sheet in excel that contians data
data = read(data_path, sheet_name='RAW')
data.head(3)

# Sort the data by the 'Date' column in ascending order
data = data.sort_values('Date')

# Group the data by 'State' and perform lag shifting within each group
grouped = data.groupby(groupby_cols)

# Create lag features within each group
for lag in lags:
    for col in lag_columns:
        data[f'{col}_lag_{lag}'] = grouped[col].shift(lag)


def week_of_month(date):
    year, month, day = map(int, date.split('-'))
    first_day = datetime.date(year, month, 1)
    adjusted_dom = first_day.weekday() + 1
    return (day + adjusted_dom - 1) // 7 + 1


data['Week_Of_Month'] = data['Date'].map(week_of_month)

# Encode categorical columns
encoder = LabelEncoder()
data['State'] = encoder.fit_transform(data['State'])

# Save the trained label encoder
joblib.dump(encoder, state_encoder)

# Split the data into features and target variable
X = data.drop(['Date', 'Anomaly', 'Sales_Amount_Upper', 'Sales_Amount_Lower'], axis=1)
y = data['Anomaly']

# Perform train-test split (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Define a narrower set of hyperparameters
params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train the XGBoost model with the reduced set of hyperparameters
model = XGBClassifier(**params, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model performance on the validation set
validation_accuracy = accuracy_score(y_val, y_val_pred)
validation_report = classification_report(y_val, y_val_pred)

print("Validation Accuracy:", validation_accuracy)
print("Validation Report:")
print(validation_report)

# Make predictions on the test set
y_test_pred = model.predict(X_test)


# Evaluate the model performance on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)

print("\nTest Accuracy:", test_accuracy)
print("Test Report:")
print(test_report)


# Load the new dataset for inference
inference_data = pd.read_csv(inference_file,
                             usecols=['Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                                      'Holiday_Flag', 'State'])

# Convert the 'Date' column to datetime format
inference_data['Date'] = pd.to_datetime(inference_data['Date']).dt.strftime('%Y-%m-%d')

# Get the saved Encoder for State column
encoder = joblib.load(state_encoder)
inference_data['State'] = encoder.fit_transform(inference_data['State'])


cols = inference_data.columns

inference_data_all = pd.concat([data[cols], inference_data])



# Sort the data by the 'Date' column in ascending order
inference_data_all = inference_data_all.sort_values('Date')

# Create lag features for the relevant columns
for lag in lags:
    for col in lag_columns:
        inference_data_all[f'{col}_lag_{lag}'] = inference_data_all[col].shift(lag)


# Get the inference data back with lag values:
inference_data = inference_data_all[inference_data_all['Date'] == '2023-10-27']
# Get week of the month value too before passing it to model
inference_data['Week_Of_Month'] = inference_data['Date'].map(week_of_month)


inference_data['Weekly_Sales'].iloc[0] = 111286394300

# Perform inference using the loaded model
predictions = model.predict(inference_data.drop(columns=['Date']))

# Display the predictions
print(predictions)
