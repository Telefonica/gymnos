#
#
#   Trainer
#
#

from dataclasses import dataclass

from ....base import BaseTrainer
from .hydra_conf import NeuralNetworkClassifierHydraConf
import pandas as pd
import numpy as np
import os
import mlflow
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score, confusion_matrix ,f1_score, precision_score, recall_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

@dataclass
class NeuralNetworkClassifierTrainer(NeuralNetworkClassifierHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):
        # Load csv file and split data (The dataset is already shuffled)
        df = pd.read_csv(root)
        X_train = df[['Age','Gender','Activity','Calories']].values[0:5000]
        Y_train = df['Alimentation'].values[0:5000]
        X_test = df[['Age','Gender','Activity','Calories']].values[6000:7000]
        Y_test = df['Alimentation'].values[6000:7000]
        X_valid = df[['Age','Gender','Activity','Calories']].values[7000:8000]
        Y_valid = df['Alimentation'].values[7000:8000]
        # Normalization
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_trasform(X_train)
        self.X_valid = self.scaler.transform(X_valid)
        self.X_test = self.scaler.transform(X_test)
        # One hot encoding on dependent variable
        self.Y_train = to_categorical(Y_train) # Its not necesary a label encoder because dependent variable is already in 0,1,2 format
        self.Y_valid = to_categorical(Y_valid)
        self.Y_test = to_categorical(Y_test)

    def train(self):
        # Neural network model
        output_neurons = len(np.unique(self.Y_train)) # Should be Y_train that has 3 different values (Y_train_NN is coded as one hot and has only 1 and 0)
        self.nn_classifer = Sequential()
        self.nn_classifer.add(Dense(self.X_train.shape[1], input_shape=(self.X_train.shape[1],), activation = self.activation_input,name = 'Input'))
        self.nn_classifer.add(Dropout( rate = self.dropout1_rate,name = 'Dropout_1'))
        self.nn_classifer.add(Dense(50, activation = self.activation_hidden1,name = 'Hidden_1'))
        self.nn_classifer.add(Dropout(rate = self.dropout2_rate,name = 'Dropout_2')) 
        # nn_classifer.add(Dense(50, activation = 'relu',name = 'Hidden_2'))
        # nn_classifer.add(Dropout(rate = 0.15,name = 'Dropout_3'))    
        self.nn_classifer.add(Dense(output_neurons, activation = self.activation_output,name = 'Output'))
        #early_stop = EarlyStopping(monitor='val_loss', min_delta=self.min_delta, patience=self.patience, verbose=self.verbose, mode='auto',restore_best_weights=True)

        # Fit model 
        self.nn_classifer.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, 
                    validation_data=(self.X_valid, self.Y_valid),
                    #callbacks=[early_stop]
                    )
        # Save model
        saving_path = os.path.join(os.getcwd(),'models','nn_classifier_cal_intake.h5')
        self.nn_classifer(saving_path)
        mlflow.log_artifact(os.getcwd(),'models','nn_classifier_cal_intake.h5')
        # Save scaler object
        saving_path = os.path.join(os.getcwd(),'models','scaler')
        joblib.dump(self.scaler,saving_path)
        mlflow.log_artifact(os.getcwd(),'models','scaler')

    def test(self):
        # Inference on test data
        preds = self.nn_classifer.predict(self.X_test)
        predicted_cal = []
        Y_test_cal = []
        for i in range(len(preds)):
            predicted_cal.append(np.argmax(preds[i]))
            Y_test_cal.append(np.argmax(self.Y_test[i])) # For taking the 1 of the one hot encoded
        # Metrics
        self.accuracy = accuracy_score(Y_test_cal, predicted_cal)
        self.precission = precision_score(Y_test_cal, predicted_cal,average = 'weighted')
        self.recall = recall_score(Y_test_cal, predicted_cal,average = 'weighted')
        self.f1_score = f1_score(Y_test_cal, predicted_cal,average = 'weighted',labels=np.unique(predicted_cal))
        self.conf_matrix = confusion_matrix(Y_test_cal,predicted_cal,labels=[0,1,2])