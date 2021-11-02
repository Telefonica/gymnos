import os
import argparse
import pandas as pd
from glob import glob
from scipy.io import wavfile


def main(args):
    """
    This function walks over the Valentini paths to create a TSV that constains all the audio metadata.
    """

    dataset_path = args.src

    # Create dataframe list
    dataframe_list = []

    # Get classes
    classes = glob(dataset_path + '/*/')
    for idx, label in enumerate(classes):
        classes[idx] = label.split('/')[-2]
    if('metadata' in classes):
        classes.remove('metadata')

    # Walk over data
    for item in classes:

        class_path = os.path.join(dataset_path, item)

        for audio_file in os.listdir(class_path):
            if(audio_file.endswith(".wav") and not audio_file.startswith('.')):

                # Sample ID
                sample_id = audio_file.split('.')[0]

                # Audio path
                audio_path = os.path.join(class_path, audio_file)

                # Audio Length
                rate, data = wavfile.read(audio_path)
                audio_length = data.shape[0]/rate

                audio_path = audio_path.replace(dataset_path, '')

                # Write row on dataframe
                dataframe_list.append([ sample_id, audio_path, audio_length, item])


    # Build valentini tsv file
    donateacry_df = pd.DataFrame(dataframe_list, columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Label'])
    donateacry_df.to_csv('donateacry.tsv', sep = '\t', index=None)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Scrip to create donateacry tsv")

    # Source Valentini data placed in the data folder of the project 
    parser.add_argument("--src", help="source directory", default="/home/fernandol/.gymnos/datasets/donateacry/donateacry_corpus_cleaned_and_updated_data/")    
    args = parser.parse_args()

    # Run main
    main(args)