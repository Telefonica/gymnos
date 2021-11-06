from os import listdir
from os.path import join
import numpy as np
from scipy import stats

def createFilenameList(root,op_list):

    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(root, target))
        samples_in_dir = [join(root, target, sample) for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)

    return [item for sublist in op_filenames for item in sublist]


def extract_features(sample, max_measurements=0, scale=1):

    features = []

    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]

    features.append(stats.median_abs_deviation(sample))
    return np.array(features).flatten()



def create_feature_set(filenames,max_measurements):
    x_out = []
    for file in filenames:
        sample = np.genfromtxt(file, delimiter=',')
        features = extract_features(sample, max_measurements)

        if len(features) >=3:
            features = features.reshape(1,3)
            x_out.append(features)


    return np.array(x_out)
