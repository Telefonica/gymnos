#
#
#   Trainer
#
#

from dataclasses import dataclass

from ....base import BaseTrainer
from .hydra_conf import ForecaasterAutoregMultiOutputHydraConf

import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import time_series_spliter
from skforecast.model_selection import cv_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import backtesting_forecaster_intervals
import matplotlib.pyplot as plt
import joblib


@dataclass
class ForecaasterAutoregMultiOutputTrainer(ForecaasterAutoregMultiOutputHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):
        df = pd.read_csv(root)
        df = df.loc[:, ['datum', product]] # Primera columna
        df['datum'] = pd.to_datetime(df['datum'], format= '%Y-%m-%d')
        #Poner la fecha como index
        df = df.set_index('datum')
        #Me quedo solo con la columna del producto que me interesa
        df = df[product]

        steps = 12
        self.df_train = df[:-steps]
        self.df_test = df[-steps:]

    def train(self):
        self.forecaster_rf = ForecasterAutoregMultiOutput(
                    regressor = Lasso(random_state=123),
                    steps     = 12,
                    lags      = 8 # Este valor será remplazado en el grid search
                )

        self.param_grid = {'alpha': np.logspace(-5, 5, 10)}

        lags_grid = [5, 12, 20]

        self.resultados_grid = grid_search_forecaster(
                                forecaster  = self.forecaster_rf,
                                y           = self.df_train,
                                param_grid  = param_grid,
                                lags_grid = lags_grid,
                                steps       = 36,
                                method      = 'cv',
                                metric      = 'mean_squared_error',
                                initial_train_size    = int(len(self.df_train)*0.5),
                                allow_incomplete_fold = False,
                                return_best = True,
                                verbose     = False
                            )
        # Save the trained model
        saving_path = os.path.join(os.getcwd(),'models','forecaster_r')
        joblib.dump(self.scaler,saving_path) # Save it also in repository
        mlflow.log_artifact(os.getcwd(),'models','forecaster_r')

    def test(self):
        steps = 12

        # Predicciones
        # ==============================================================================
        predicciones = self.forecaster_rf.predict(steps=steps)
        # Se añade el índice temporal a las predicciones
        predicciones = pd.Series(data=predicciones, index=self.df_test.index)
        predicciones.head()


        # Gráfico
        # ==============================================================================
        fig, ax = plt.subplots(figsize=(9, 4))
        self.df_train.plot(ax=ax, label='train')
        #s2_test.plot(ax=ax, label='test')
        predicciones.plot(ax=ax, label='predicciones')
        ax.legend()
