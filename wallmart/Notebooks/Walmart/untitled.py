from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def handle_dates(data, cols):
    # Convert Date type column to datetime
    for date in cols:
        data[date] = pd.to_datetime(data[date])

    return data
    

def handle_categoricals_lable_encode(data, cols):
    # Encode the categorical columns column
    for col in cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    return data

def handle_categoricals_onehot_encode(data, cols):
    # One-hot encode the 'State' column
    encoder = OneHotEncoder(sparse=False)
    encoded_states = encoder.fit_transform(data[['State']])
    
    # Create a DataFrame with the encoded state columns, name columns as states
    encoded_states_df = pd.DataFrame(encoded_states, columns=encoder.get_feature_names(['State']))
    
    

def split_data(data_prepared):
    # Define X (features) and y (target)
    X = data_prepared.drop('Anomaly', axis=1)
    y = data_prepared['Anomaly']
    
    # Split the dataset into train and test sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Further split the train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    print("X_train.shape, X_val.shape, X_test.shape", X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test

def main(data):
    data = handle_dates(data, ['Date'])
    data = handle_categoricals_lable_encode(data, ['State'])

    # Concatenate the encoded states dataframe with the original data
    # Drop the original 'State' column and 'Date' column as we won't use 'Date' in model training directly
    data = pd.concat([data.drop(['State', 'Date'], axis=1), encoded_states_df], axis=1)

    
    X_train, X_val, X_test = split_data(data)



