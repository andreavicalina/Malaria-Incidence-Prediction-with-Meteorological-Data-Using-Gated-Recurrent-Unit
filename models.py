import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from keras.layers import Dense, GRU, Dropout, Activation
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint,  EarlyStopping
import streamlit as st
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam, Adamax, SGD, RMSprop
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
import re


tf.config.run_functions_eagerly(True)


def preprocessing_iklim_test(df):
    df = pd.read_csv(df, error_bad_lines=False)
    df = pd.DataFrame(df)

    df[['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car', 'week', 'year', 'weeks']
       ] = df['Tanggal'].apply(lambda x: re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', x, maxsplit=13)).apply(pd.Series)

    # Convert 'Tanggal' column to datetime format using the correct format
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['weeks'] = df['Tanggal'].dt.to_period('W').dt.start_time

    # fill missing values with 0
    df['Tavg'] = df['Tavg'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['RH_avg'] = df['RH_avg'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['RR'] = df['RR'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['ss'] = df['ss'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['ff_x'] = df['ff_x'].replace([8888, 9999, 8888.0, 9999.0], 0.0)

    columns_to_convert = ['Tavg',  'RH_avg', 'RR', 'ss', 'ff_x']
    df[columns_to_convert] = df[columns_to_convert].replace(
        {',': '.'}, regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(
        pd.to_numeric, errors='coerce')

    df_weather = df.drop(['Tanggal', 'week', 'year'], axis=1)

    df['Tavg'] = df['Tavg'].fillna(df['Tavg'].mean())
    df['RH_avg'] = df['RH_avg'].fillna(df['RH_avg'].mean())
    mean_RR = df['RR'].mean()
    df['RR'] = df['RR'].fillna(float(mean_RR))
    df['ss'] = df['ss'].fillna(df['ss'].mean())
    df['ff_x'] = df['ff_x'].fillna(df['ff_x'].mean())

    df_weather = df_weather.groupby(['weeks']).mean()
    st.sidebar.write("_" * 30)

    return df_weather


def preprocessing_iklim(df):
    df = pd.read_csv(df, error_bad_lines=False)
    df = pd.DataFrame(df)

    # Convert 'Tanggal' column to datetime format using the correct format
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['weeks'] = df['Tanggal'].dt.to_period('W').dt.start_time

    # fill missing values with 0
    df['Tavg'] = df['Tavg'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['RH_avg'] = df['RH_avg'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['RR'] = df['RR'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['ss'] = df['ss'].replace([8888, 9999, 8888.0, 9999.0], 0.0)
    df['ff_x'] = df['ff_x'].replace([8888, 9999, 8888.0, 9999.0], 0.0)

    columns_to_convert = ['Tavg',  'RH_avg', 'RR', 'ss', 'ff_x']
    df[columns_to_convert] = df[columns_to_convert].replace(
        {',': '.'}, regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(
        pd.to_numeric, errors='coerce')

    df_weather = df.drop(['Tanggal', 'week', 'year'], axis=1)

    df['Tavg'] = df['Tavg'].fillna(df['Tavg'].mean())
    df['RH_avg'] = df['RH_avg'].fillna(df['RH_avg'].mean())
    mean_RR = df['RR'].mean()
    df['RR'] = df['RR'].fillna(float(mean_RR))
    df['ss'] = df['ss'].fillna(df['ss'].mean())
    df['ff_x'] = df['ff_x'].fillna(df['ff_x'].mean())

    df_weather = df_weather.groupby(['weeks']).mean()
    st.sidebar.write("_" * 30)

    return df_weather


def preprocessing_malaria(df_case):
    df_case = pd.read_csv(df_case,  error_bad_lines=False)
    df_case['weeks'] = pd.to_datetime(df_case[['year', 'month']].assign(
        day=1)) + pd.to_timedelta(df_case['week'] * 7, unit='D')
    df_case['weeks'] = df_case['weeks'].dt.to_period('W').dt.start_time
    df_case['weeks'] = pd.to_datetime(df_case['weeks'])

    df_case.insert(1, column='total_case', value=1)
    df_case = df_case.groupby(['weeks'], as_index=False).sum()

    return df_case


def preprocess_file_iklim(file):
    try:
        uploaded_file1 = preprocessing_iklim(uploaded_file1)
    except Exception as e:
        uploaded_file1 = preprocessing_iklim_test(uploaded_file1)

    return file


def merging(uploaded_file1, uploaded_file2):
    try:
        uploaded_file = preprocessing_iklim(uploaded_file1)
        uploaded_file1 = uploaded_file

    except Exception as e:
        uploaded_file = preprocessing_iklim_test(uploaded_file1)
        uploaded_file1 = uploaded_file

    uploaded_file2 = preprocessing_malaria(uploaded_file2)
    merged_df = pd.merge(uploaded_file1, uploaded_file2,
                         on=['weeks'], how='left')
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.sort_values(by=['weeks'], ascending=True)
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df[['weeks', 'total_case',
                           'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x']]
    return merged_df


def merging_test(uploaded_file1, uploaded_file2):
    uploaded_file1 = preprocessing_iklim_test(uploaded_file1)
    uploaded_file2 = preprocessing_malaria(uploaded_file2)
    merged_df = pd.merge(uploaded_file1, uploaded_file2,
                         on=['weeks'], how='left')
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.sort_values(by=['weeks'], ascending=True)
    merged_df = merged_df.reset_index(drop=True)
    # merged_df = merged_df[['weeks', 'total_case',
    #                        'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x']]
    merged_df = merged_df[['weeks', 'total_case', 'Tavg', 'RR', 'ff_x']]
    return merged_df


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_predict(dataset, n_in, n_out=1, n_vars=3, train_proportions=1):
    # read data
    dataset = pd.DataFrame(dataset)
    # Setel indeks timestamp
    dataset['weeks'] = pd.to_datetime(dataset['weeks'])
    dataset.set_index("weeks", inplace=True)
    values = dataset.values
    # ubah semua values ke float
    values = values.astype('float32')
    # normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # mengubah data series menjadi supervised
    reframed = series_to_supervised(scaled, n_in, n_out)
    # hapus variabel yang dicadangkan
    contain_vars = []
    for i in range(1, n_in+1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars+1)]
    data = reframed[contain_vars +
                    ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1, n_out)]]
    # ubah nama kolom
    col_names = ['Y', 'X1', 'X2', 'X3']
    contain_vars = []
    for i in range(n_vars):
        contain_vars += [('%s(t-%d)' % (col_names[i], j))
                         for j in range(1, n_in+1)]
    data.columns = contain_vars + \
        ['Y(t)'] + [('Y(t+%d)' % (j)) for j in range(1, n_out)]

    values = data.values

    n_train = int((len(values) * train_proportions)+1)

    test = values[n_train:, :]
    test_X, test_y = test[:, :n_in*n_vars], test[:, n_in*n_vars:]
    test_X = test_X.reshape((test_X.shape[0], n_in, n_vars))

    val = values[:n_train:, :]
    val_x, val_y = val[:, :n_in*n_vars], val[:, n_in*n_vars:]
    val_x = val_x.reshape((val_x.shape[0], n_in, n_vars))

    return scaler, data, val_x, val_y, test_X


def prepare_data(dataset, n_in, n_out=1, n_vars=3, train_proportions=0.8):
    # read data
    dataset = pd.DataFrame(dataset)
    # Setel indeks timestamp
    dataset['weeks'] = pd.to_datetime(dataset['weeks'])
    dataset.set_index("weeks", inplace=True)
    values = dataset.values
    # ubah semua values ke float
    values = values.astype('float32')
    # normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # mengubah data series menjadi supervised
    reframed = series_to_supervised(scaled, n_in, n_out)
    # hapus variabel yang dicadangkan
    contain_vars = []
    for i in range(1, n_in+1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars+1)]
    data = reframed[contain_vars +
                    ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1, n_out)]]
    # ubah nama kolom
    col_names = ['Y', 'X1', 'X2', 'X3']
    contain_vars = []
    for i in range(n_vars):
        contain_vars += [('%s(t-%d)' % (col_names[i], j))
                         for j in range(1, n_in+1)]
    data.columns = contain_vars + \
        ['Y(t)'] + [('Y(t+%d)' % (j)) for j in range(1, n_out)]
    # split data
    values = data.values
    n_train = int((len(values) * train_proportions)+1)
    # n_train = 157
    train = values[:n_train, :]
    test = values[n_train:, :]
    # pisahkan input x dan output y
    train_X, train_y = train[:, :n_in*n_vars], train[:, n_in*n_vars:]
    test_X, test_y = test[:, :n_in*n_vars], test[:, n_in*n_vars:]
    # ubah input x ke dalam format lstm [samples,timesteps,features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_vars))

    # n_val = int((len(values) * (1-train_proportions))+1)
    val = values[n_train:, :]
    val_x, val_y = val[:, :n_in*n_vars], val[:, n_in*n_vars:]

    val_x = val_x.reshape((val_x.shape[0], n_in, n_vars))

    return scaler, data, train_X, train_y, test_X, test_y, val_x, val_y, dataset


def fit_gru(data_prepare, n_neurons, n_batch, n_epoch, loss='mse', optimizer='Adam', l2_reg=0.01):
    train_X = data_prepare[2]
    train_y = data_prepare[3]
    test_X = data_prepare[4]
    test_y = data_prepare[5]

    # Design GRU architecture
    model = Sequential()
    model.add(GRU(units=64, input_shape=(
        train_X.shape[1], train_X.shape[2]), kernel_regularizer=tf.keras.regularizers.l2(l2_reg), return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # model.add(Dense(train_y.shape[1]))
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)

    model.summary()

    checkpoint = ModelCheckpoint(
        'best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch,
                        validation_data=(test_X, test_y),
                        verbose=1, shuffle=False, callbacks=[checkpoint, early_stopping])

    return model, history


def train_and_save_best_model(data_prepare, n_neuron, n_batch, n_epoch, repeats, learning_rates):
    best_val_loss = float('inf')  # Initialize with a high value
    best_model = None
    combined_training_loss = []  # List to store combined training loss values
    combined_validation_loss = []  # List to store combined validation loss values
    model_list = []  # Initialize an empty list to store models

    for i in range(repeats):
        # Ambil learning rate untuk iterasi saat ini
        current_learning_rate = learning_rates

        # Panggil fungsi fit_gru dengan learning rate yang sesuai
        model, history = fit_gru(data_prepare, n_neuron, n_batch, n_epoch, optimizer=Adam(
            learning_rate=current_learning_rate))
        model_list.append(model)

        # Store the training and validation loss values in lists
        combined_training_loss.extend(history.history['loss'])
        combined_validation_loss.extend(history.history['val_loss'])

        # Update the best model if the current model has lower validation loss
        if history.history['val_loss'][-1] < best_val_loss:
            best_val_loss = history.history['val_loss'][-1]
            best_model = model

    # Save the best trained model using ModelCheckpoint
    best_model.save('best_model.h5')

    return best_model, combined_training_loss, combined_validation_loss, model_list


def gru_predict(model, data_prepare, yhat_prev=None):
    scaler = data_prepare[0]
    test_x = data_prepare[4]
    # test_y = data_prepare[5]

    # Jika yhat_prev ada, gunakan yhat_prev, jika tidak, lakukan prediksi baru
    if yhat_prev is None:
        yhat = model.predict(test_x)
    else:
        yhat = yhat_prev

    # retransfrom data prediksi ke data asli
    scale_new = MinMaxScaler()
    scale_new.min_, scale_new.scale_ = scaler.min_[0], scaler.scale_[0]
    inv_yhat = scale_new.inverse_transform(yhat)

    # kembalikan nilai aktual ke data asli
    # inv_y = scale_new.inverse_transform(test_y)
    return inv_yhat, yhat


def gru_predict_test(model, data_prepare, yhat_prev=None):
    scaler = data_prepare[0]
    test_x = data_prepare[2]
    # test_y = data_prepare[5]

    # Jika yhat_prev ada, gunakan yhat_prev, jika tidak, lakukan prediksi baru
    if yhat_prev is None:
        yhat = model.predict(test_x)
    else:
        yhat = yhat_prev

    # retransfrom data prediksi ke data asli
    scale_new = MinMaxScaler()
    scale_new.min_, scale_new.scale_ = scaler.min_[0], scaler.scale_[0]
    inv_yhat = scale_new.inverse_transform(yhat)

    # kembalikan nilai aktual ke data asli
    # inv_y = scale_new.inverse_transform(test_y)
    return inv_yhat, yhat
