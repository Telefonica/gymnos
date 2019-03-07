import os
import logging
import subprocess
import zipfile


class KaggleBase(object):
    '''
        Works as a service for kaggle data sets 
    '''
    def __init__(self):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "KAGGLE_BASE"
        
    def download(self, localDir, sourceType, sourceId):
        self._datasetLocalDir = localDir
        os.makedirs(localDir)
        cmd = ['kaggle',  sourceType, 'download', '-c', sourceId, '-p', localDir]
        subprocess.call(cmd)
        self.__uncompressDataSetFiles()

    def __uncompressDataSetFiles(self):
        extension = ".zip"
        for item in os.listdir(self._datasetLocalDir):
            if item.endswith(extension): 
                file_name = os.path.join(self._datasetLocalDir, item) 
                self._log.info("{0} - Unzipping {1} ...".format(self._log_prefix, file_name))
                zip_ref = zipfile.ZipFile(file_name) 
                zip_ref.extractall(self._datasetLocalDir) 
                zip_ref.close() 
                os.remove(file_name) 
