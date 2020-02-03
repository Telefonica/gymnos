#
#
#  Repetition
#
#

import logging

import numpy as np
import pandas as pd

from .dataset import Dataset, Array, ClassLabel

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

ENTITIES = "ENTITIES"
LABEL_COL_NAME = "HAS_RETRY"
UTTERANCCES_COL_NAME = "INPUTS"


class Repetition(Dataset):
    """
    This class generates a repetition dataset that consists of dataset with one feature of sequences of phrases
    of Aura logs in spoken in one session and, as binary class label, if the sequences are a repetition of an implicit
    action that not was solved by Aura in the session.

    Firstly, loads:
     - text samples contain 1 text attribute consists of list of list of phrases in string format from Aura logs.

    After that, makes parses several characters in utterances column:
    - Replaces in label column commas by points.
    - Drops nan rows.
    - Replaces multiple null spaces in sequences
    - Parses nan includes in every sequence.
    - Fills with ones all values upper or equal than label_threshold parameter al with zeros the rest of values
    - Parses compose entities in sequences with underscores
    - Remove sequences with only brackets.

    In the end, the class labels will be:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | no repetition|
    +----------+--------------+
    | 1        | repetition   |
    +----------+--------------+

    -  Characteristics
        - **Classes**: 2
        - **Samples total**: variable
        - **Features**: texts

    Parameters
    ===========
    path_input_name: str
        Path to input file
    embedding_names: list of strings,
        Trained Embeddings names (.ppkl)
    label_threshold : float,
        Threshoold applied to label to determine if upper or equalthan it is a repetition (1) and otherwise not (0).
    """

    def __init__(self, path_input_name, label_threshold=0.5):
        self.path_input_name = path_input_name
        self.label_threshold = label_threshold

    @property
    def features_info(self):
        return Array(shape=[1], dtype=str)

    @property
    def labels_info(self):
        return ClassLabel(names=["not_repetition", "repetition"])

    def download_and_prepare(self, dl_manager):
        # load data
        df = pd.read_csv(self.path_input_name, sep=",")

        # Parses several characters in utterances column
        df = self.__parsing(df)

        self.labels_ = df[LABEL_COL_NAME].values
        self.features_ = df[UTTERANCCES_COL_NAME].values

    def __parsing(self, df):
        """
        Parses a dataframe in high level applying:
        - Replaces in label column commas by points.
        - Drops nan rows.
        - Replaces multiple null spaces in sequences
        - Parses nan includes in every sequence.
        - Fills with ones all values upper or equal than label_threshold parameter al with zeros the rest of values
        - Parses compose entities in sequences with underscores
        - Remove sequences with only brackets.

        Parameters
        ----------
        df: dataframe
            Features and label

        Returns
        --------
        df: dataframe
            Features and label parsed.
        """

        # Replaces in label column commas by points.
        df[LABEL_COL_NAME] = [float(str(val).replace(",", ".")) for val in df[LABEL_COL_NAME].values]

        # Drops nan
        df = df.dropna()

        # Replaces multiple null spaces in sequences
        df[UTTERANCCES_COL_NAME] = df[UTTERANCCES_COL_NAME].str.replace(' ' + str(np.nan) + ' ', '', regex=True)

        # Parses nan include in every sequence
        df = Repetition.__parsing_expressions(data=df, utterances_col_name=UTTERANCCES_COL_NAME,
                                              type_parsing="global_nan")

        # Fills with ones all values upper or equal than label_threshold parameter al with zeros the rest of values
        df[LABEL_COL_NAME][df[LABEL_COL_NAME] >= self.label_threshold] = 1.0
        df[LABEL_COL_NAME][df[LABEL_COL_NAME] < self.label_threshold] = 0.0

        # Parses compose entities in sequences with underscores
        total_utterances_col_name = []
        list_entities = [eval(val) for val in df[ENTITIES].values]
        for pos_sequence, sequence in enumerate(df[UTTERANCCES_COL_NAME].values):
            sequence = Repetition.__parsing_composed_entities(sequence, list_entities[pos_sequence])
            sequence = Repetition.__parsing_expressions(data=sequence, utterances_col_name=UTTERANCCES_COL_NAME,
                                                        type_parsing="sequence_others")
            total_utterances_col_name.append(sequence)
        df[UTTERANCCES_COL_NAME] = total_utterances_col_name

        # Removes sequences with only brackets
        df = df[df[UTTERANCCES_COL_NAME].str.len() > 2]

        # Drops nan
        df = df.dropna()
        return df

    @staticmethod
    def __parsing_expressions(data, utterances_col_name, type_parsing, iteration_global_nan=8):
        """
        Parses a dataframe in high level applying:
            - If type="global_nan", parses the total column of utterrances erasing nan.
            - If type="sequence_others", parses a sequence from utterances column with other characters.

        Parameters
        ----------
        data: dataframe or string
            Features and label
        type_parsing: string,
            Type of parsing ("global_nan": apply to the whole utterance column to erase nan
            or "sequence_others": apply to sequences of utterances column to to erase other characters)
        iteration_global_nan: int, optional
            Number of iteration of parsing with type="global_nan"

        Returns
        --------
        self: Preprocessor
            Own instance for chain purposes.
        """
        # TODO use regular expression to simplify parsing
        assert type_parsing in ("global_nan", "sequence_others")
        if type_parsing == "global_nan":
            dict_strings_to_parse_nan = {r"\[" + str(np.nan): '[',
                                         r"\[ " + str(np.nan): '[',
                                         str(np.nan) + r"\]": ']',
                                         str(np.nan) + r" \]": ']',
                                         str(np.nan) + "'": "'",
                                         "'" + str(np.nan): "'",
                                         "\n" + str(np.nan): "",
                                         str(np.nan) + "\n": ""}
            for k, v in dict_strings_to_parse_nan.items():
                for cont in range(iteration_global_nan):
                    data[utterances_col_name] = data[utterances_col_name].str.replace(k, v, regex=True)

        elif type_parsing == "sequence_others":
            dict_strings_to_parse_sequence = {"\r": "",
                                              "\n": "",
                                              "''": "','",
                                              "' '": "','",
                                              "[ ]": "[]",
                                              "[  ]": "[]",
                                              "[   ]": "[]",
                                              "[    ]": "[]",
                                              "[     ]": "[]",
                                              "[      ]": "[]",
                                              "[       ]": "[]",
                                              "[         ]": "[]",
                                              "[          ]": "[]",
                                              "[           ]": "[]"}
            for k, v in dict_strings_to_parse_sequence.items():
                data = data.replace(k, v)
        else:
            pass
        return data

    @staticmethod
    def __parsing_composed_entities(sequence, list_entities):
        """
        Parses compose entities in sequences with underscores.

        Parameters
        ----------
        sequence: string,
            Input sequence of utterances
        list_entities: string,
            Entities detected in sequence.

        Returns
        --------
        sequence: string,
            parsed sequence.
        """
        list_entities_sequence = [item for sublist in list_entities for item in sublist]
        for entity in list_entities_sequence:
            if sequence.__contains__(entity):
                if len(entity.split(" ")) >= 2:
                    sequence = sequence.replace(entity, '_'.join(entity.split(" ")))
        return sequence

    def __getitem__(self, index):
        return self.features_[index], self.labels_[index]

    def __len__(self):
        return len(self.features_)
