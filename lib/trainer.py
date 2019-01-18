import os, time, h5py, logging, inspect, json, importlib, progressbar
import numpy as np
import tensorflow as tf
import keras
#import models

from datetime import datetime
from dataset_manager import DataSetManager
from session_manager import SessionManager
from model_manager import ModelManager

BASE_PATH = '/home/sysadmin/aitp/'
SYS_CONFIG_PATH = BASE_PATH + 'config/system.json'

with open(SYS_CONFIG_PATH, 'rb') as fp:
  sys_config = json.load(fp)

TRAINING_EXECUTION_PATH = sys_config['paths']['training']['execution']
DATASETS_PATH = sys_config['paths']['datasets']

class Trainer(object):
    def __init__(self, config):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "TRAINER"
        self._log.info("{0} - Initializing...".format(self._log_prefix))
        self._log.info("{0} - Configuration received - {1} ".format(self._log_prefix, json.dumps(config, indent=4, sort_keys=True)))
        self._config = config
        self._dataSetId = self._config["dataset"]["id"]
        dsmConfig = {key: value for (key, value) in (self._config["dataset"].items() + self._config["training"].items())}
        self._dsm = DataSetManager(dsmConfig)
        self._sm = SessionManager(self._config["session"])
        self._mm = ModelManager(self._config["model"])


    def run(self):
       self.prepareTrainingSession()
       self.executeTraining()
       self.generateTrainingStats()


    def prepareTrainingSession(self):
        self._trainingId = '{0}_{1}_{2}_{3}_{4}'.format( self._config['problem']['source'],
                                                         self._config['problem']['id'],
                                                         self._config['model']['id'],
                                                         self._config['dataset']['id'],
                                                         self._config['hyper_params']['learning_strategy'] )
        self._log.info("{0} - prepareTrainingSession - generated id - {1}".format(self._log_prefix, self._trainingId))
        self.__checkIfTestAlreadyExecuted()
        self._trainingTimeStamp = time.strftime("%Y%m%d-%H%M%S")
        self._train_dir = '{0}/{1}/{2}'.format( TRAINING_EXECUTION_PATH,
                                                self._trainingId,
                                                self._trainingTimeStamp )
        self.__createTrainingDirectory()
        self.__loadDataSet()
        self.__loadModel()
        
    
    def executeTraining(self):
        self._log.info("{0} - Training starts ...".format(self._log_prefix))       
        start = datetime.now()
        history = self._model.fit( self._fitSamples, 
                                   self._fitLabels, 
                                   epochs=self._config['hyper_params']['epochs'],
                                   batch_size=self._config['hyper_params']['batch_size'],
                                   validation_data=(self._valSamples, self._valLabels),
                                   verbose=1 )
        end = datetime.now()
        elapsed = end - start
        self._log.info("{0} - Training completed in - {1} (s): {2} (us)".format( self._log_prefix, 
                                                                          elapsed.seconds,
                                                                          elapsed.microseconds,)) 
        trainedWeightsPath = '{0}/weights.h5'.format(self._train_dir)
        self._log.info("{0} - Saving weights at - {1}".format( self._log_prefix, trainedWeightsPath ))
        model.save(trainedWeightsPath)

 
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

    def __loadTrainingSamples(self):
        self._log.debug("{0} - __loadTrainingSamples - obtaining training samples...".format(self._log_prefix))
        (self._fitSamples, self._valSamples, self._testSamples) = self._dsm.getSamplesForTraining()
        (self._fitLabels, self._valLabels, self._testLabels) = self._dsm.getLabelsForTraining()
        
    def __loadDataSet(self):
        self._log.info("{0} - Loading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._dsm.loadDataSet()
        self.__loadTrainingSamples()
       
    def __loadModel(self):
        self._model = self._mm.getModel()
        self._model.init()
        if self._model.checkFineTunning() is True:
            self._log.info("{0} - Fine tunning required.".format(self._log_prefix))
            self.__loadTrainingSamples()
            (flattenFit, flattenVal, flattenTest) = self._model.fineTune( self._fitSamples, 
                                                                          self._valSamples, 
                                                                          self._testSamples,
                                                                          self._fitLabels, 
                                                                          self._valLabels, 
                                                                          self._testLabels,
                                                                          self._train_dir )
            # Update training samples with flatten values
            self._fitSamples = flattenFit
            self._valSamples = flattenVal
            self._testSamples = flattenTest
            
        self._model.compile()
        self._model.summary()

        

    '''

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
'''
