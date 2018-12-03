import os, time, h5py, logging, inspect, json
import numpy as np
import tensorflow as tf
import keras
import keras.datasets
from keras.preprocessing.image import ImageDataGenerator
import models
from dataset_downloader import *

BASE_PATH = '/home/sysadmin/aitp/'
SYS_CONFIG_PATH = BASE_PATH + 'config/system.json'

with open(SYS_CONFIG_PATH, 'rb') as fp:
  sys_config = json.load(fp)

TRAINING_EXECUTION_PATH = sys_config['paths']['training']['execution']
DATASETS_PATH = sys_config['paths']['datasets']

class Trainer(object):

    MODEL_LIST_OFFSET_ID         = 0
    MODEL_LIST_OFFSET_TARGET     = 1

    _modelList = {

        # KEY                ID      TARGET             DESCRIPTION
        # ------------------------------------------------------------------------------
        'vgg16':           [ 100,   'vgg16'       ],   # Model based on VGG16
        'vgg19':           [ 200,   'VGG19'       ],   # Model based on VGG19
        'xception':        [ 300,   'Xception'    ],   # Model based on Xception
        'resnet50':        [ 400,   'ResNet50'    ],   # Model based on ResNet50
        'inceptionv3':     [ 500,   'InceptionV3' ]    # Model based on InceptionV3
    }
    

    def __init__(self, config):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "TRAINER"
        self._log.info("{0} - Initializing...".format(self._log_prefix))
        self._log.info("{0} - Configuration received - {1} ".format(self._log_prefix, json.dumps(config, indent=4, sort_keys=True)))
        self._config = config


    def __checkIfTestAlreadyExecuted(self):
        targetDir = '{0}/{1}'.format(TRAINING_EXECUTION_PATH, self._trainingId)
        self._log.debug("{0} - __checkIfTestAlreadyExecuted - training id - {1}".format(self._log_prefix, self._trainingId))
        if os.path.isdir(targetDir):
            self._log.warning("{0} - Training already executed.".format(self._log_prefix))
        else:
            self._log.info("{0} - Training never executed before.".format(self._log_prefix))

    def __createTrainingDirectory(self):
        if not os.path.exists(self._train_dir): 
            os.makedirs(self._train_dir)
            self._log.debug("{0} - __createTrainingDirectory - training directory created at - {1}".format(self._log_prefix, self._train_dir))

    def __loadModelFromLibrary(self):
        modelId = self._config["model"]["id"]
        self._log.debug("{0} - __loadModelFromLibrary - looking for model id - {1}".format(self._log_prefix, modelId))
        for name, data in inspect.getmembers(models):
            if name == '__builtins__':
                continue
            if name == modelId:
                self._log.error("{0} - Model {1} supported.".format(self._log_prefix, modelId))
                target = self._modelList[modelId][self.MODEL_LIST_OFFSET_TARGET]
                self._model = globals()[target]()
        if self._model is None:
            self._log.error("{0} - Model {1} not supported.".format(self._log_prefix, modelId))
        

    def __loadPretrainedWeights(self):
        pass
        # lookup and load


    def __dataSetInKeras(self):
        retval = False
        if not self._dataSetId in inspect.getmembers(keras.datasets):
            self._log.warning("{0} - Data set not found in keras.".format(self._log_prefix))
        else:
            self._log.info("{0} - Data set found in keras.".format(self._log_prefix))
            retval = True

        return retval


    def __dataSetInLocalVolume(self):
        retval = False
        targetDir = '{0}/{1}'.format(DATASETS_PATH, self._dataSetId)
        if os.path.isdir(targetDir):
            self._log.info("{0} - Data set found in local volume.".format(self._log_prefix))
        else:
            self._log.warning("{0} - Data set not found in local volume.".format(self._log_prefix))

        return retval


    def __loadDataSet(self):
        self._dataSetId = self._config["dataset"]["id"]
        if self.__dataSetInKeras():
            self.__loadDataSetFromKeras()
        else:
            if self.__dataSetInLocalVolume():
                self.__loadImagesFromLocalDataSet()
            else:
                self.__downloadDataSet()
            

    def __loadDataSetFromKeras(self):
        self._dataSet = globals()[self._dataSetId]()
        (x_train, y_train), (x_test, y_test) = self._dataSet.load_data()
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)


    def __loadImagesFromLocalDataSet(self):        
        train_dir = '{0}/{1}/train'.format(DATASETS_PATH, self._dataSetId)
        val_dir = '{0}/{1}/val'.format(DATASETS_PATH, self._dataSetId)
        test_dir = '{0}/{1}/test'.format(DATASETS_PATH, self._dataSetId)

        train_datagen = ImageDataGenerator(rescale=1./255)       # Pixel Normalization
        val_datagen = ImageDataGenerator(rescale=1./255)         # Pixel Normalization
        test_datagen = ImageDataGenerator(rescale=1./255)        # Pixel Normalization

        imgage_width = self._config.dataset.properties.image_width
        imgage_heigth = self._config.dataset.properties.imgage_heigth
        batch_size = self._config.dataset.properties.batch_size
        class_mode = self._config.dataset.properties.class_mode

        self._train_generator = train_datagen.flow_from_directory( train_dir, 
                                                                   target_size=(imgage_width, imgage_heigth),
                                                                   batch_size=batch_size,
                                                                   class_mode=class_mode )

        self._val_generator = val_datagen.flow_from_directory( val_dir,
                                                               target_size=(imgage_width, imgage_heigth),
                                                               batch_size=batch_size,
                                                               class_mode=class_mode )

        self._test_generator = val_datagen.flow_from_directory( test_dir,
                                                                target_size=(imgage_width, imgage_heigth),
                                                                batch_size=batch_size,
                                                                class_mode=class_mode )

    def __downloadDataSet(self):
        pass

    def __loadTextFromLocalDataSet(self): 
        pass


    def __loadDataSetFromLocal(self, dataSetId):
        dataSetType = self._config.dataset.type
        if dataSetType == 'image':
            self.__loadImagesFromLocalDataSet(dataSetId)
        elif dataSetType == 'text':
            self.__loadTextFromLocalDataSet(dataSetId)


    def prepareTrainingSession(self):
        self._trainingId = '{0}_lr_{1}_update_G{2}'.format( self._config['dataset']['id'],
                                                            self._config['hyper_params']['learning_rate'],
                                                            self._config['hyper_params']['update_rate'] )
        self._log.info("{0} - prepareTrainingSession - generated id - {1}".format(self._log_prefix, self._trainingId))
        self.__checkIfTestAlreadyExecuted()
        self._trainingTimeStamp = time.strftime("%Y%m%d-%H%M%S")
        self._train_dir = '{0}/{1}/{2}'.format( TRAINING_EXECUTION_PATH,
                                                self._trainingId,
                                                self._trainingTimeStamp )
        self.__createTrainingDirectory()
        self.__loadModelFromLibrary()
        self.__loadPretrainedWeights()
        self.__loadDataSet()

       
    def run(self):
       self.prepareTrainingSession()
       self.executeTraining()
       self.generateTrainingStats()



       '''

        # --- checkpoint and monitoring ---
        all_vars = tf.trainable_variables()

        d_var = [v for v in all_vars if v.name.startswith('Discriminator')]
        log.warn("********* d_var ********** "); slim.model_analyzer.analyze_vars(d_var, print_info=True)

        g_var = [v for v in all_vars if v.name.startswith(('Generator'))]
        log.warn("********* g_var ********** "); slim.model_analyzer.analyze_vars(g_var, print_info=True)

        rem_var = (set(all_vars) - set(d_var) - set(g_var))
        print([v.name for v in rem_var]); assert not rem_var

        self.d_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.d_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate*0.5,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='d_optimize_loss',
            variables=d_var
        )

        self.g_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='g_optimize_loss',
            variables=g_var
        )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=100)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor =  tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)

        max_steps = 1000000

        output_save_step = 1000
        test_sample_step = 100

        for s in xrange(max_steps):
            step, accuracy, summary, d_loss, g_loss, s_loss, step_time, prediction_train, gt_train, g_img = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            # periodic inference
            if s % test_sample_step == 0:
                accuracy_test, prediction_test, gt_test = \
                    self.run_test(self.batch_test, is_train=False)
            else:
                accuracy_test = 0.0

            if s % 10 == 0:
                self.log_step_message(step, accuracy, accuracy_test, d_loss, g_loss, s_loss, step_time)

            self.summary_writer.add_summary(summary, global_step=step)

            if s % output_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                save_path = self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)
                if self.config.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'g_img_'+str(s)+'.hy'), 'w')
                    f['image'] = g_img
                    f.close()

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op, self.model.d_loss, self.model.g_loss,
                 self.model.S_loss, self.model.all_preds, self.model.all_targets, self.model.fake_img, self.check_op]

        if step%(self.config.update_rate+1) > 0:
        # Train the generator
            fetch.append(self.g_optimizer)
        else:
        # Train the discriminator
            fetch.append(self.d_optimizer)

        fetch_values = self.session.run(fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, loss, summary, d_loss, g_loss, s_loss, all_preds, all_targets, g_img] = fetch_values[:9]

        _end_time = time.time()

        return step, loss, summary, d_loss, g_loss, s_loss,  (_end_time - _start_time), all_preds, all_targets, g_img

    def run_test(self, batch, is_train=False, repeat_times=8):

        batch_chunk = self.session.run(batch)

        [step, loss, all_preds, all_targets] = self.session.run(
            [self.global_step, self.model.accuracy, self.model.all_preds, self.model.all_targets],
            feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False))

        return loss, all_preds, all_targets

    def log_step_message(self, step, accuracy, accuracy_test, d_loss, g_loss, s_loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Supervised loss: {s_loss:.5f} " +
                "D loss: {d_loss:.5f} " +
                "G loss: {g_loss:.5f} " +
                "Accuracy: {accuracy:.5f} "
                "test loss: {test_loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step = step,
                         d_loss = d_loss,
                         g_loss = g_loss,
                         s_loss = s_loss,
                         accuracy = accuracy,
                         test_loss = accuracy_test,
                         sec_per_batch = step_time,
                         instance_per_sec = self.batch_size / step_time
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--update_rate', type=int, default=5)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--dump_result', action='store_true', default=False)
    config = parser.parse_args()

    if config.dataset == 'MNIST':
        import  datasets.mnist as dataset
    elif config.dataset == 'SVHN':
        import datasets.svhn as dataset
    elif config.dataset == 'CIFAR10':
        import datasets.cifar10 as dataset
    else:
        raise ValueError(config.dataset)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    config.deconv_info = dataset.get_deconv_info()
    dataset_train, dataset_test = dataset.create_default_splits()

    trainer = Trainer(config,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f", config.dataset, config.learning_rate)
    trainer.train()
    '''
