import models as ml
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from matplotlib.dates import DateFormatter
from scipy.interpolate import interp1d
import seaborn as sns
from math import sqrt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

tf.config.run_functions_eagerly(True)


matplotlib.use('Agg')


st.header(
    'Application of Gated Recurrent Unit to Predict Malaria Incidence using Meteorological Data')

st.markdown('---')

# --------------------------------------SHOWING DATA (set sidebar)-----------------------------------------#
#### Training Data ####
st.sidebar.header('1. Training Data')
# data iklim
uploaded_file1 = st.sidebar.file_uploader(
    label="Please upload the meteorological data (csv) for training process",
    type="csv",
    accept_multiple_files=False)

# data kasus malaria
uploaded_file2 = st.sidebar.file_uploader(
    label="Please upload the malaria incidence data (csv) for training process",
    type="csv",
    accept_multiple_files=False)

button_train = st.sidebar.button('Train')
st.sidebar.markdown('---')

st.sidebar.header('2. Testing Data')
# Get user input for testing data
testing1 = st.sidebar.file_uploader(
    "Please upload the meteorological data (csv) for testing process", type=["csv"])
testing2 = st.sidebar.file_uploader(
    "Please upload the malaria incidence data (csv) for testing process", type=["csv"])
button_testing = st.sidebar.button('Test')
st.sidebar.markdown('---')

st.sidebar.header('3. Predicting Data')
# Get user input for testing data
prediction1 = st.sidebar.file_uploader(
    "Please upload the meteorological data (csv) for predicting process", type=["csv"])
prediction2 = st.sidebar.file_uploader(
    "Please upload the malaria incidence data (csv) for predicting process", type=["csv"])
button_predict = st.sidebar.button('Predict')


# --------------------------------------Closing Sidebar--------------------------------------#

if uploaded_file1 and uploaded_file2 is not None:
    if button_train:
        merged_df = ml.merging(uploaded_file1, uploaded_file2)
        st.write("""
        ### Data
        """)
        merged_df['weeks'] = pd.to_datetime(merged_df['weeks'])

        plot_data = merged_df[merged_df['weeks'].dt.year <= 2021]

        st.write(
            "This is the integration dataset that will be used in training process")
        st.dataframe(plot_data)

        n_in = 52
        n_out = 1
        n_neuron = 64
        n_batch = 8
        n_epoch = 500
        repeats = 1
        n_vars = 3

        data_prepare = ml.prepare_data(merged_df, n_in, n_out)
        scaler, data, train_X, train_y, test_X, test_y, val_x, val_y, dataset = data_prepare

        best_model, training_loss, validation_loss, models = ml.train_and_save_best_model(
            data_prepare, n_neuron, n_batch, n_epoch, repeats, learning_rates=0.001)

        # Display the plot of the combined training and validation loss
        st.write('Training Loss dan Validation Loss Plot')
        # Display the plot of the training loss
        plt.figure(figsize=(10, 6))

        # Subplot 1 for training loss
        plt.subplot(2, 1, 1)
        plt.plot(training_loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Subplot 2 for validation loss
        plt.subplot(2, 1, 2)
        plt.plot(validation_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.tight_layout()  # Ensures proper spacing between subplots

        plt.savefig('loss_plot.png')
        st.image('loss_plot.png', use_column_width=True)

        # Show the score of training and validation loss
        st.write('Training Loss: %.3f' % training_loss[-1])
        st.write('Validation Loss: %.3f' % validation_loss[-1])

elif testing1 and testing2 is not None:
    if button_testing:
        n_in = 52
        n_out = 1

        loaded_model = load_model('best_model.h5')
        # Preprocess and prepare testing data
        merged_testing = ml.merging(testing1, testing2)
        data_prepare_testing = ml.prepare_data(
            merged_testing, n_in, n_out)

        # Initialize yhat_prev to None
        yhat_prev = None

        # Perform prediction using the loaded model
        inv_yhat, yhat = ml.gru_predict(
            loaded_model, data_prepare_testing, yhat_prev=None)

        # retransfrom data prediksi ke data asli
        scaler = data_prepare_testing[0]
        scale_new = MinMaxScaler()
        scale_new.min_, scale_new.scale_ = scaler.min_[0], scaler.scale_[0]

        inv_y = scale_new.inverse_transform(data_prepare_testing[5])

        mae = mean_absolute_error(inv_y, inv_yhat)
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

        #### Prediction ####
        rounded_inv_yhat = np.round(inv_yhat).astype(int)
        rounded_inv_y = np.round(inv_y).astype(int)

        rounded_inv_yhat[rounded_inv_yhat < 0] = 0

        # Pastikan DataFrame sudah diurutkan berdasarkan indeks (jika belum)
        merged_testing_sorted = merged_testing.sort_index()

        # Mengambil 10% indeks terakhir dari kolom yang diinginkan (misalnya, kolom 'nama_kolom')
        weeks_data = merged_testing_sorted['weeks'].tail(
            len(rounded_inv_yhat[:, 0]))

        comparison_df = pd.DataFrame({
            'weeks': weeks_data,
            'predicted': rounded_inv_yhat[:, 0],
            # Hanya ambil kolom pertama dari nilai aktual
            'actual': rounded_inv_y[:, 0]
        })
        comparison_df = comparison_df.reset_index(drop=True)
        st.subheader('Table of Predicted vs Actual Cases')
        st.write(
            'This table contains the comparison values of predicted and actual cases')
        st.table(comparison_df)

        # Display the comparison DataFrame
        st.subheader('Actual vs Predicted Plot')
        st.write('This plot shows the comparison between predicted and actual cases')

        # Generate more data points with weekly intervals for smoother lines
        weeks_extended = pd.date_range(
            start=comparison_df['weeks'].min(), end=comparison_df['weeks'].max(), freq='D')

        # Interpolate the data using scipy's interp1d
        f_predicted = interp1d(comparison_df['weeks'].astype(
            int), comparison_df['predicted'])
        f_actual = interp1d(comparison_df['weeks'].astype(
            int), comparison_df['actual'], kind='cubic')

        # Create a DataFrame with the extended weeks and interpolated values
        interpolated_df = pd.DataFrame({'weeks': weeks_extended})
        interpolated_df['predicted'] = f_predicted(
            interpolated_df['weeks'].astype(int))
        interpolated_df['actual'] = f_actual(
            interpolated_df['weeks'].astype(int))

        # Create a smoother line chart using Seaborn with a smooth line style
        plt.figure(figsize=(20, 10))
        ax = sns.lineplot(data=interpolated_df, x='weeks',
                          y='predicted')
        sns.lineplot(data=interpolated_df, x='weeks',
                     y='actual', ci=None)

        # Customize line style
        # Solid line for predicted
        ax.lines[0].set_linestyle("-")
        ax.lines[0].set_label("Predicted")
        # Solid line for actual
        ax.lines[1].set_linestyle("-")
        ax.lines[1].set_label("Actual")

        # Plot data points with markers at weekly intervals
        sns.scatterplot(data=comparison_df, x='weeks',
                        y='predicted', marker='o', color='blue')
        sns.scatterplot(data=comparison_df, x='weeks',
                        y='actual',  marker='o', color='orange')

        # Add labels and title
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()

        # Save and display the plot
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig('testing.png')
        st.image("testing.png", width=900)

        st.subheader("Evaluation Model Performance:")
        st.write(
            "To evaluate the performance of the model, we will use the following metrics:")
        st.write("**Mean Absolute Error (MAE):**")
        st.write('%.3f' % mae)
        st.write("**Root Mean Squares Error (RMSE):**")
        st.write('%.3f' % rmse)


# prediction
elif prediction1 and prediction2 is not None:
    if button_predict:
        n_in = 52
        n_out = 1
        n_vars = 3

        loaded_model = load_model('best_model.h5')

        # Preprocess and prepare prediction data

        merged_predict = ml.merging(prediction1, prediction2)

        data_prepare_predict = ml.prepare_predict(merged_predict, n_in, n_out)

        # Assuming your prediction data is stored in test_X
        # Use the modified prepare_predict_data result
        # val_x = data_prepare_predict[6]
        val_x = data_prepare_predict[2]

        # Now, you can use this reshaped test_X to make predictions
        inv_yhat, yhat = ml.gru_predict_test(
            loaded_model, data_prepare_predict)

        # Post-process the predictions
        rounded_inv_yhat = np.round(inv_yhat).astype(int)
        rounded_inv_yhat[rounded_inv_yhat < 0] = 0

        # Create a DataFrame to display the predictions
        prediction_df = pd.DataFrame({
            'Number of Weeks': range(1, rounded_inv_yhat[:12, 0].shape[0]+1),
            'Predicted Cases': rounded_inv_yhat[:12, 0]
        })
        # Display the prediction DataFrame
        st.subheader("Predicted Result")
        st.write("Predicted Malaria Cases on 12 weeks ahead:")
        st.table(prediction_df)

        # Create a line plot
        st.subheader('Plot of Predicted Cases')
        st.write('This plot shows the predicted cases over time')

        # Generate more data points with finer granularity for smoother curves
        weeks_extended = np.linspace(
            prediction_df['Number of Weeks'].min(),
            prediction_df['Number of Weeks'].max(),
            100  # Adjust the number of data points for smoother curves
        )

        # Interpolate the data using scipy's interp1d
        f_predicted = interp1d(
            prediction_df['Number of Weeks'],
            prediction_df['Predicted Cases'],
            kind='cubic'
        )

        # Create a DataFrame with the extended weeks and interpolated values
        interpolated_df = pd.DataFrame({'Number of Weeks': weeks_extended})
        interpolated_df['Predicted Cases'] = f_predicted(
            interpolated_df['Number of Weeks']
        )

        # Create a smoother line chart using Seaborn with a smooth line style
        plt.figure(figsize=(10, 6))

        # Plot the smooth curve
        sns.lineplot(
            data=interpolated_df,
            x='Number of Weeks',
            y='Predicted Cases',
            # marker=False,  # Do not show markers for the smooth curve
            ci=None
        )

        # Plot data points with markers based on the original weekly data
        sns.scatterplot(
            data=prediction_df,
            x='Number of Weeks',
            y='Predicted Cases',
            marker='o',  # Show markers for data points
            color='blue',
            alpha=0.7,  # Adjust the transparency for a cleaner appearance
        )

        # Customize line style for the smooth curve
        # Solid line for the smooth curve
        plt.gca().lines[0].set_linestyle("-")

        # Add labels and title
        plt.xlabel('Weeks')
        plt.ylabel('Cases')
        plt.title('Predicted Cases Over Time (Smoothed)')
        plt.legend()

        # Save and display the plot
        plt.savefig('prediction.png')

        # Display the larger image with a width of 1000 pixels
        st.image('prediction.png', width=1000)

else:
    st.info('Awaiting for CSV file of data climate and malaria cases to be uploaded.')
