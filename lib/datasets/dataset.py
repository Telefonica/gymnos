import os, h5py, progressbar
import numpy as np

class DataSet(object):
    def __init__(self):
        pass
    
    def prepare_h5py(self, shape=None):

        image = np.concatenate((self._trainImages, self._testImages), axis=0).astype(np.uint8)
        label = np.concatenate((self._trainLabels, self._trainLabels), axis=0).astype(np.uint8)

        print('Preprocessing data...')

        bar = progressbar.ProgressBar( maxval=100,
                                        widgets=[progressbar.Bar('=', '[', ']'),
                                        ' ', 
                                        progressbar.Percentage()] )
        bar.start()

        f = h5py.File(os.path.join(self._datasetLocalDir, 'data.hy'), 'w')
        data_id = open(os.path.join(self._datasetLocalDir,'id.txt'), 'w')

        for i in range(image.shape[0]):
            if i%(image.shape[0]/100)==0:
                bar.update(i/(image.shape[0]/100))
            grp = f.create_group(str(i))
            data_id.write(str(i)+'\n')
            if shape:
                grp['image'] = np.reshape(image[i], shape, order='F')
            else:
                grp['image'] = image[i]
            label_vec = np.zeros(10)
            label_vec[label[i]%10] = 1
            grp['label'] = label_vec.astype(np.bool)

        bar.finish()
        f.close()
        data_id.close()
        
        return