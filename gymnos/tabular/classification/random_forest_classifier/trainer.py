#
#
#   Trainer
#
#

from dataclasses import dataclass

from ....base import BaseTrainer
from .hydra_conf import RandomForestClassifierHydraConf
import pandas as pd
import numpy as np
import os
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix ,f1_score, precision_score, recall_score


@dataclass
class RandomForestClassifierTrainer(RandomForestClassifierHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):
        # Load csv file and split data (The dataset is already shuffled)
        df = pd.read_csv(root)
        self.X_train = df[['Age','Gender','Activity','Calories']].values[0:5000]
        self.Y_train = df['Alimentation'].values[0:5000]
        self.X_test = df[['Age','Gender','Activity','Calories']].values[6000:7000]
        self.Y_test = df['Alimentation'].values[6000:7000]
        self.X_valid = df[['Age','Gender','Activity','Calories']].values[7000:8000]
        self.Y_valid = df['Alimentation'].values[7000:8000]

    def train(self):
        # Define random forest model
        self.rf_model = RandomForestClassifier(random_state=42)
        self.rf_model.set_params(n_estimators=self.n_trees, max_features=self.max_features)
        # Train the model with data
        self.rf_model.fit(self.X_train,self.Y_train)
        # Save the trained model
        saving_path = os.path.join(os.getcwd(),'models','rf_model')
        joblib.dump(self.scaler,saving_path) # Save it also in repository
        mlflow.log_artifact(os.getcwd(),'models','rf_model')

    def test(self):
        # Inference on test data
        preds = self.rf_model.predict(self.X_test)
        # Metrics
        self.accuracy = accuracy_score(self.Y_test, preds)
        self.precission = precision_score(self.Y_test, preds,average = 'weighted')
        self.recall = recall_score(self.Y_test, preds,average = 'weighted')
        self.f1_score = f1_score(self.Y_test, preds,average = 'weighted',labels=np.unique(preds))
        self.conf_matrix = confusion_matrix(self.Y_test,preds,labels=[0,1,2])
