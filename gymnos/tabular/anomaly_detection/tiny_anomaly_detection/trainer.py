#
#
#   Trainer
#
#

from dataclasses import dataclass
import logging
from ....base import BaseTrainer
from .hydra_conf import TinyAnomalyDetectionHydraConf
from .utils import createFilenameList, extract_features, create_feature_set, convert_to_tiny
from .model import create_tiny_model
from .c_writer import create_array


@dataclass
class TinyAnomalyDetectionTrainer(TinyAnomalyDetectionHydraConf, BaseTrainer):

    def prepare_data(self, root):

        logger = logging.getLogger(__name__)
        # Defining sample directories
        dataset_path = 'arduino'
        normal_op_list = ['normal']
        anomaly_op_list = ['anomal']
        # Creating variables needed for the training
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio
        sensor_sample_rate = self.sensor_sample_rate
        sample_time = self.sample_time
        samples_per_file = self.samples_per_file
        max_measurements = int(sample_time * sensor_sample_rate)
        # Creating a list of samples
        normal_op_filenames = createFilenameList(root, normal_op_list)
        anomaly_op_filenames = createFilenameList(root, anomaly_op_list)

        val_set_size = int(len(normal_op_filenames) * val_ratio)
        test_set_size = int(len(normal_op_filenames) * test_ratio)

        # Splitting into train, val and test datasets
        num_samples = len(normal_op_filenames)
        filenames_val = normal_op_filenames[:val_set_size]
        filenames_test = normal_op_filenames[val_set_size:(
            val_set_size + test_set_size)]
        filenames_train = normal_op_filenames[(val_set_size + test_set_size):]

        logger.info('Number of training samples: ' + str(len(filenames_train)))
        logger.info('Number of validation samples: ' + str(len(filenames_val)))
        logger.info('Number of test samples: ' + str(len(filenames_test)))

        logger.info('Creating sets...')
        # Extracting features to feed the model
        self.x_train = create_feature_set(filenames_train, max_measurements)

        self.x_val = create_feature_set(filenames_val, max_measurements)

        self.x_test = create_feature_set(filenames_test, max_measurements)

        self.x_train = self.x_train.reshape((len(self.x_train), 3))
        self.sample_shape = (self.x_train.shape[1:])
        self.directory = root

    def train(self):

        logger = logging.getLogger(__name__)
        logger.info('Creating model...')
        self.model = create_tiny_model(
            self.encoding_dim, self.dropout, self.sample_shape)
        self.model.summary()

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

        logger.info('Start training')
        history = self.model.fit(self.x_train,
                                 self.x_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.x_val, self.x_val),
                                 verbose=1)

        logger.info('End training')
        self.model.save(
            self.directory + "/" + self.model_name + '.h5')

        # Converting models to fit on MCUs
        convert_to_tiny(self)

    def test(self):

        logger = logging.getLogger(__name__)

        # Calculate MSE from validation set
        predictions = self.model.predict(self.x_val)
        normal_mse = np.mean(np.power(self.x_val - predictions, 2), axis=1)
        logger.info('Average MSE for normal validation set: ' +
                    str(np.average(normal_mse)))
        logger.info('Standard deviation of MSE for normal validation set: ' +
                    str(np.std(normal_mse)))
        logger.info('Recommended threshold (3x std dev + avg): ' +
                    str((3 * np.std(normal_mse)) + np.average(normal_mse)))

        self.model.save(root + self.model_name + '.h5')
