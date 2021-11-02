import os
import json
import argparse
import pandas as pd

from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit


def generate_labels_column(dataframe, classes_index):

    inv_classes_index = {v: k for k, v in classes_index.items()}
    dataframe.insert(len(dataframe.columns)-1, "Label_Index", "")
    
    for index, row in dataframe.iterrows():
        label = row['Label']
        dataframe.at[index, 'Label_Index'] = inv_classes_index[label]

    return dataframe


def generate_partition_information(train_df, dev_df, test_df):

    # Get metrics about the number of samples
    sample_number_train = len(train_df.index)
    sample_number_dev = len(dev_df.index)
    sample_number_test = len(test_df.index)
    sample_number_total = sample_number_train + sample_number_dev + sample_number_test

    sample_percentage_train = (sample_number_train/sample_number_total)*100
    sample_percentage_dev = (sample_number_dev/sample_number_total)*100
    sample_percentage_test = (sample_number_test/sample_number_total)*100

    # Get hours information
    audio_hours_wuw_train = (train_df["Audio_Length"].sum())/3600.0
    audio_hours_wuw_dev = (dev_df["Audio_Length"].sum())/3600.0
    audio_hours_wuw_test = (test_df["Audio_Length"].sum())/3600.0
    audio_hours_wuw_total = audio_hours_wuw_train + audio_hours_wuw_dev + audio_hours_wuw_test

    # Sound Type Information information
    samples_info_train = train_df['Label'].value_counts().to_dict()
    samples_info_dev = dev_df['Label'].value_counts().to_dict()
    samples_info_test = test_df['Label'].value_counts().to_dict()
    samples_info_total = Counter(samples_info_train)
    samples_info_total.update(samples_info_dev)
    samples_info_total.update(samples_info_test)
    samples_info_total = dict(samples_info_total)

    information = {
        'General': {
            'Number of samples' : sample_number_total,
            'Hours of audio' : audio_hours_wuw_total,
            'Samples information': samples_info_total
        },
        'Train': {
            'Number of samples' : sample_number_train,
            'Hours of audio' : audio_hours_wuw_train,
            'Sample percentage' : sample_percentage_train,
            'Samples information' : samples_info_train
        },
        'Validation': {
            'Number of samples' : sample_number_dev,
            'Hours of audio' : audio_hours_wuw_dev,
            'Sample percentage' : sample_percentage_dev,
            'Samples information' : samples_info_dev
        },
        'Test': {
            'Number of samples' : sample_number_test,
            'Hours of audio' : audio_hours_wuw_test,
            'Sample percentage' : sample_percentage_test,
            'Noise' : samples_info_test
        }
    }

    return information


def check_low_quantity_labels(dataframe):
    labels = dataframe['Label'].value_counts().to_dict()
    print(labels)
    low_quantity_data = [k for k, v in labels.items() if v == 1]

    if(len(low_quantity_data) > 0):
        low_quantity_data_df = dataframe[dataframe['Label'].isin(low_quantity_data)]
        dataframe = dataframe[~dataframe['Label'].isin(low_quantity_data)]
        return dataframe, low_quantity_data_df
    else:
        return dataframe


def get_stratified_partition(dataframe, seed):
    dataframe = check_low_quantity_labels(dataframe)
    train_df, dev_df = split_dataset(dataframe, test_size=0.1, random_state=seed)
    dev_df, low_quantity_data_df = check_low_quantity_labels(dev_df)
    dev_df, test_df = split_dataset(dev_df, test_size=0.5, random_state=seed)
    test_df = test_df.append(low_quantity_data_df)
    return train_df, dev_df, test_df


def split_dataset(dataframe, test_size=0.2, random_state=0):
    train_inds, test_inds = next(StratifiedShuffleSplit(
        test_size=test_size,
        n_splits=2,
        random_state=random_state
        ).split(dataframe, dataframe['Label'].tolist()))
    train_df = dataframe.iloc[train_inds]
    test_df = dataframe.iloc[test_inds]

    return train_df, test_df


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


def get_classes_index(dataframe):
    """
    Get a class index dependeding on the unique classes existing in the dataset.
    """
    unique_values = list(dataframe['Label'].unique())
    classes_index = {}

    for i in range(len(unique_values)):
        classes_index[i] = unique_values[i]

    return classes_index


def main(args):

    # Destination path
    if(args.dst !=  ""):
        output_path = args.dst
    else:
        output_path = os.path.join(args.src, 'metadata')
    check_path(output_path)

    # Read tsv
    df_path = os.path.join(args.src, 'metadata/donateacry.tsv')
    df = pd.read_csv(df_path, header=0, sep='\t')

    # Get and save classes index
    classes_index = get_classes_index(df)
    with open(output_path + '/classes_index.json', 'w', encoding='utf-8') as f:
        json.dump(classes_index, f, ensure_ascii=False, indent=4)

    # Generate Label index column
    df = generate_labels_column(df, classes_index)

    # Generate partition
    train_df, dev_df, test_df = get_stratified_partition(df, args.seed)

    # Generate partition information
    partition_info = generate_partition_information(train_df, dev_df, test_df)

    # Save partitions information
    with open(output_path + '/partition.json' , 'w', encoding='utf-8') as f:
            json.dump(partition_info, f, ensure_ascii=False, indent=4)

    # Save partitions
    train_df.to_csv(os.path.join(output_path, "train.tsv"), sep='\t', index=None)
    dev_df.to_csv(os.path.join(output_path, "dev.tsv"), sep='\t', index=None)
    test_df.to_csv(os.path.join(output_path, "test.tsv"), sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to partitions for donateacry dataset")

    parser.add_argument("--src", help="source directory", default="/home/fernandol/.gymnos/datasets/donateacry/donateacry_corpus_cleaned_and_updated_data/") 
    parser.add_argument("--dst", help="destination directory", default="")   
    parser.add_argument('--seed', type=int, default=0, help='partition seed')
    args = parser.parse_args()

    main(args)