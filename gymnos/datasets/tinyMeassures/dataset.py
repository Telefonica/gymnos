#
#
#   Tiny Meassures dataset
#
#


import logging
import os
from ...base import BaseDataset
from .hydra_conf import TinyMeassuresHydraConf
from ...utils.data_utils import extract_archive
from dataclasses import dataclass
from ...services.sofia import SOFIA
from .processing import split_in_csv


@dataclass
class TinyMeassures(TinyMeassuresHydraConf, BaseDataset):

    """
    The data consists on TXT with all the data obtain for a two days interval.
    It will be splitted on two directories, anomal and normal. Each of them will contain
    a set of CSV files with three columns constant meassures from
    the arduino sensors. They are in the form of temperature*humidity*preassure
    Split must be done using *.

    Assumptions
    -----------

    After some data analysis, we have decided to consider temperatures over 32º
    as anomalies. As it is unsupervised learning, the model may learn other thresholds so
    It could be interesting to play arrounf with that temperature threshold.

        """

    def download(self, root):

        logger = logging.getLogger(__name__)
        download_dir = SOFIA.download_dataset(
            "IndIAna_jones/datasets/Arduino_environmental_data")
        logger.info("Extracting some magic powder ...")

        # Reading txt with all meassures
        arduino_data = open(download_dir + '/arduino_meassures.txt', 'r')
        Lines = arduino_data.readlines()

        # Lists to hold the meassures
        temp = []
        humidity = []
        pres = []
        temp_anomal = []
        humidity_anomal = []
        preassure_anomal = []

        for line in Lines:
            meassurements = line.split("*")
            try:  # avoiding meassurements errors
                if len(meassurements) == 3:

                    if float(meassurements[0].strip()) >= 33:  # Anomaly threshold
                        temp_anomal.append(meassurements[0].strip())
                        humidity_anomal.append(meassurements[1].strip())
                        preassure_anomal.append(meassurements[2].strip())
                    else:
                        temp.append(meassurements[0].strip())  # deleting \n
                        humidity.append(meassurements[1].strip())
                        pres.append(meassurements[2].strip())

            except Exception as e:
                continue

        logger.info("Preparing samples")
        logger.info("It will take a little")
        logger.info("Grab some coffee ☕")
        # spltting samples in csv files
        split_in_csv(temp, humidity, pres, root, "normal")
        logger.info("normal samples prepared")
        split_in_csv(temp_anomal, humidity_anomal,
                     preassure_anomal, root, "anomal")
        logger.info("All samples prepared! :D")
