import os, time, logging, inspect, json, importlib, progressbar
import numpy as np
import tensorflow as tf
import keras

from datetime import datetime
from dataset_manager import DataSetManager
from session_manager import SessionManager
from model_manager import ModelManager
from callback_provider import CallbackProvider

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
        self._cp = CallbackProvider(self._config["training"]["callbacks"])

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
        self.__saveOriginalConfigToFile()
        self.__loadDataSet()
        self.__loadModel()
        
    def executeTraining(self):
        self._log.info("{0} - Training starts ...".format(self._log_prefix))       
        start = datetime.now()
        self._cp.buildCallbackList({"train_dir": self._train_dir})
        self._history = self._model.fit( self._fitSamples, 
                                         self._fitLabels, 
                                         epochs=self._config['hyper_params']['epochs'],
                                         batch_size=self._config['hyper_params']['batch_size'],
                                         validation_data=(self._valSamples, self._valLabels),
                                         callbacks=self._cp.getList(),
                                         verbose=1 )
        end = datetime.now()
        elapsed = end - start
        self._log.info("{0} - Training completed in - {1} (s): {2} (us)".format( self._log_prefix, 
                                                                          elapsed.seconds,
                                                                          elapsed.microseconds,)) 

        if self._cp.isCallbackPresentInList("model_checkpoint") is False:                                                                       
            trainedWeightsPath = '{0}/weights.h5'.format(self._train_dir)
            self._log.info("{0} - Saving weights by default at - {1}".format( self._log_prefix, trainedWeightsPath ))
            self._model.save(trainedWeightsPath)
 
    def generateTrainingStats(self):
        self.__showTrainingHistory()
        self.__evaluateModel()
        
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

    def __evaluateModel(self):
        self._log.info("{0} - Evaluating model with:\n[\n\t - test_samples = {1}\
                                                    \n\t - test_labels = {2}\
                                                    \n]".format( self._log_prefix,
                                                                 self._testSamples.shape, 
                                                                 self._testLabels.shape ))
        self._results = self._model.evaluate(self._testSamples, self._testLabels)     
                        
        self._log.info("{0} - Evaluation results:\n[\n\t - loss = {1}\
                                                    \n\t - accuracy = {2}\
                                                    \n]".format( self._log_prefix,
                                                                 self._results[0], 
                                                                 self._results[1] ))

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

    def __saveOriginalConfigToFile(self):
        self._log.debug("{0} - __saveOriginalConfigToFile - saving original config to train directory...".format(self._log_prefix))
        with open('{0}/experiment_original_config.json'.format(self._train_dir), 'w') as outfile:
            json.dump(self._config, outfile, indent=4, sort_keys=True)

    def __showTrainingHistory(self):
        self._log.info("{0} - Training was executed with:\n[\n\t - metrics = {1}\
                                                            \n\t - samples = {2}\
                                                            \n\t - epochs = {3}\
                                                            \n\t - steps = {4}\
                                                            \n\t - validation = {5}\
                                                            \n]".format( self._log_prefix,
                                                                         self._history.params["metrics"],
                                                                         self._history.params["samples"],
                                                                         self._history.params["epochs"],
                                                                         self._history.params["steps"],
                                                                         self._history.params["do_validation"] ))

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

'''
