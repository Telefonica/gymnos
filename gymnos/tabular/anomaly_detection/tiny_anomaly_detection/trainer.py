#
#
#   Trainer
#
#

from dataclasses import dataclass
import logging
from ....base import BaseTrainer
from .hydra_conf import TinyAnomalyDetectionHydraConf
from .utils import  createFilenameList,extract_features,create_feature_set
from .model import create_tiny_model

@dataclass
class TinyAnomalyDetectionTrainer(TinyAnomalyDetectionHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):

        logger = logging.getLogger(__name__)
        dataset_path = 'arduino'
        normal_op_list = ['normal']
        anomaly_op_list = ['anomal']
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio
        sensor_sample_rate = self.sensor_sample_rate
        sample_time = self.sample_time
        samples_per_file = self.samples_per_file
        max_measurements = int(sample_time * sensor_sample_rate)

        normal_op_filenames = createFilenameList(root,normal_op_list)
        anomaly_op_filenames = createFilenameList(root,anomaly_op_list)

        val_set_size = int(len(normal_op_filenames) * val_ratio)
        test_set_size = int(len(normal_op_filenames) * test_ratio)

        num_samples = len(normal_op_filenames)
        filenames_val = normal_op_filenames[:val_set_size]
        filenames_test = normal_op_filenames[val_set_size:(val_set_size + test_set_size)]
        filenames_train = normal_op_filenames[(val_set_size + test_set_size):]

        logger.info('Number of training samples: '+ str(len(filenames_train)))
        logger.info('Number of validation samples: '+ str(len(filenames_val)))
        logger.info('Number of test samples: '+str(len(filenames_test)))

        # Create training, validation, and test sets
        x_train = create_feature_set(filenames_train, max_measurements)

        x_val = create_feature_set(filenames_val, max_measurements)

        x_test = create_feature_set(filenames_test, max_measurements)

        x_train = x_train.reshape((len(x_train),3))


    def train(self):

        logger.info('Creating model')
        model = create_tiny_model(self.encoding_dim,self.dropout)
        model.summary()


        model.compile(optimizer=sefl.optimizer,
             loss=self.loss)
        logger.info('Start training')
        history = model.fit(x_train,
                   x_train,
                   epochs=self.epochs,
                   batch_size=self.batch_size,
                   validation_data=(x_val, x_val),
                   verbose=1)
        logger.info('End training')




    def test(self):
        pass   # OPTIONAL: test code
