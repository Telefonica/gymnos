#
#
#   Media Tagging Engine
#
#

import os
import requests
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

from ..logger import get_logger
from .dataset import PublicDataset
from ..utils.io_utils import read_from_json


GENRE_TO_SUBSCRIPTION_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "resources", "mte",
                                                  "genre_to_subscription.json")


class MTE(PublicDataset):
    """
    Dataset to predict topics of video contents from M+ based on the title and the description of the content.

    The class labels are:

    +----------+----------------------------+
    | Label    | Description                |
    +==========+============================+
    | 0        | Deportes                   |
    +----------+----------------------------+
    | 1        | Salud y Belleza            |
    +----------+----------------------------+
    | 2        | Humor                      |
    +----------+----------------------------+
    | 3        | Hogar                      |
    +----------+----------------------------+
    | 4        | Cine                       |
    +----------+----------------------------+
    | 5        | Cultural y Educativo       |
    +----------+----------------------------+
    | 6        | Infantil                   |
    +----------+----------------------------+
    | 7        | Música                     |
    +----------+----------------------------+
    | 8        | Entretenimiento            |
    +----------+----------------------------+
    | 9        | Información y Actualidad   |
    +----------+----------------------------+
    | 10       | Documental                 |
    +----------+----------------------------+
    | 11       | Tecnología                 |
    +----------+----------------------------+
    | 12       | Moda                       |
    +----------+----------------------------+
    | 13       | Viajes                     |
    +----------+----------------------------+
    | 14       | Serie                      |
    +----------+----------------------------+
    | 15       | Motor                      |
    +----------+----------------------------+
    | 16       | Cocina                     |
    +----------+----------------------------+

    Characteristics
        - **Classes**: 17 (multilabel)
        - **Samples total**: variable
        - **Features**: texts
    """

    public_dataset_files = ("http://ottcache.dof6.com/movistarplus/webplayer.hls/OTT/epg?from={}&span=7&channel=" +
                            "&network=movistarplus").format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)

        self.logger = get_logger(prefix=self)

    def read(self, download_dir):
        epgs_file_paths = [os.path.join(download_dir, epg) for epg in os.listdir(download_dir)]
        epg_df = self.__parse_epgs(epgs_file_paths)
        epg_df = self.__clean_epg(epg_df)
        epg_df["subscriptions"] = self.__map_genre_to_subscriptions(epg_df.genre)

        epg_df = epg_df[epg_df.subscriptions.notna()]

        X = epg_df.title + " " + epg_df.description

        y = epg_df.subscriptions
        y = self.__convert_to_multilabel(y)

        return X, y

    def __convert_to_multilabel(self, subscriptions):
        mlb = MultiLabelBinarizer()

        return mlb.fit_transform(subscriptions)


    def __clean_epg(self, df):
        df = df.drop_duplicates()
        df = df.fillna({"title": ""})
        df = df[df != ""]
        df = df[df.genre != "SIN CLASIFICAR"]
        df = df.dropna(subset=["genre"])
        df = df.dropna(subset=["description"])

        return df


    def __map_genre_to_subscriptions(self, genres):
        mapping_json = read_from_json(GENRE_TO_SUBSCRIPTION_MAPPING_PATH)
        subscriptions = genres.map(mapping_json)

        if subscriptions.isna().sum() > 0:
            unknown_genres = genres[subscriptions.isna()].unique().tolist()
            self.logger.warning("Unknown subscription for the following genres: {}".format(unknown_genres))

        return subscriptions


    def __safe_download_json(self, url, params=None):
        try:
            res = requests.get(url, params=params)
            return res.json()
        except (requests.exceptions.RequestException, ValueError) as err:
            self.logger.warning("{}: {}".format(url, err))
            return None


    def __parse_epgs(self, epgs_file_paths):
        df = pd.DataFrame()

        for epg_file_path in tqdm(epgs_file_paths, leave=False, desc="EPGs"):
            epg_df = self.__parse_epg(epg_file_path)
            df = pd.concat([df, epg_df], ignore_index=True)

        return df


    def __parse_epg(self, file_path):
        epg_json = read_from_json(file_path)

        df = pd.DataFrame(columns=["title", "description", "genre", "channel_id"])

        df_idx = 0

        for channel in tqdm(epg_json, leave=False, desc="Channels"):
            for program in tqdm(channel, desc="Programs"):
                sheet_url = program.get("Ficha")
                if sheet_url is None:
                    continue

                sheet = self.__safe_download_json(sheet_url)
                if sheet is None:
                    continue

                if not isinstance(sheet, dict):
                    self.logger.warning("Error downloading {}: {}".format(sheet_url, sheet))
                    continue

                df.loc[df_idx] = self.__parse_sheet(sheet)

                df_idx += 1

        return df


    def __parse_sheet(self, sheet):
        title = sheet.get("Titulo")
        description = sheet.get("Descripcion")

        try:
            channel_id = sheet["Pases"][0]["Canal"]["CodCadenaTv"]
        except (KeyError, IndexError):
            channel_id = None

        genre = sheet.get("Genero", {}).get("ComAntena")

        return (title, description, genre, channel_id)
