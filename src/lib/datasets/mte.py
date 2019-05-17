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
from .dataset import ClassificationDataset
from .mixins import PublicURLMixin
from ..utils.io_utils import read_from_json


GENRE_TO_SUBSCRIPTION = {
    "ATLETISMO": [1],
    "BALONCESTO": [1],
    "BALONMANO": [1],
    "BELLEZA": [2],
    "BRICOLAJE": [4],
    "CAZA Y PESCA": [1],
    "CICLISMO": [1],
    "CINE": [5],
    "CINE ADULTO": [5],
    "CINE CIENCIA FICCIÓN": [5],
    "CINE COMEDIA": [5, 3],
    "CINE COMEDIA ROMANT.": [5, 3],
    "CINE DE ACCIÓN": [5],
    "CINE DE ANIMACIÓN": [5],
    "CINE DE AVENTURAS": [5],
    "CINE DRAMA": [5],
    "CINE HISTÓRICO": [5, 6],
    "CINE INFANTIL": [5, 7],
    "CINE MUSICAL": [5, 8],
    "CINE OESTE": [5],
    "CINE POLICIACO": [5],
    "CINE SUSPENSE": [5],
    "CINE TERROR": [5],
    "CINE TV": [5],
    "COCINA": [17],
    "CONCIERTO": [8],
    "CONCURSO": [9],
    "CORAZÓN / SOCIEDAD": [9],
    "CORTO": [],
    "CULTURAL/EDUCATIVO": [6],
    "DANCE / ELECTRÓNICA": [8],
    "DANZA / BALLET": [8],
    "DEBATE": [10],
    "DECORACIÓN": [4],
    "DEPORTE": [1],
    "DEPORTES": [1],
    "DEPORTE ACUÁTICO": [1],
    "DEPORTE DE INVIERNO": [1],
    "DEPORTE TRADICIONAL": [1],
    "DIBUJOS ANIMADOS": [7],
    "DOC. ACTUALIDAD": [11],
    "DOC. ARTE Y CULTURA": [11, 6],
    "DOC. CAZA Y PESCA": [11, 1],
    "DOC. CIENCIA Y TEC.": [11, 12],
    "DOC. NATURALEZA": [11],
    "DOC. TENDENCIAS": [11, 13],
    "DOC. VIAJES": [11, 14],
    "DOCU-SERIE": [11, 15],
    "DOCUMENTAL": [11],
    "DOCUMENTAL BIOGRAFÍA": [11],
    "DOCUMENTAL DE CINE": [11, 5],
    "DOCUMENTAL HISTORIA": [11, 6],
    "DOCUMENTAL MÚSICA": [11, 8],
    "DOCUMENTAL POLÍTICA": [11, 10],
    "DOCUMENTAL SOCIEDAD": [11],
    "ECONOMÍA": [10],
    "ENTRETENIM. DEBATE": [9],
    "ENTRETENIM. ENTREVISTA": [9],
    "ENTRETENIMIENTO": [9],
    "ENTREVISTA": [9, 10],
    "ESOTERISMO": [],
    "FORMACIÓN ACADÉMICA": [6],
    "FÚTBOL": [1],
    "FÚTBOL AMERICANO": [1],
    "GOLF": [1],
    "HOCKEY": [1],
    "HUMOR": [3],
    "IDIOMAS": [6],
    "INF. SOCIEDAD": [],
    "INFANTIL": [7],
    "INFANTIL EDUCATIVO": [7],
    "INFORMACIÓN": [10],
    "INFORMACIÓN DEPORTE": [10, 1],
    "INFORMACIÓN POLÍTICA": [10],
    "INFORMATIVO": [10],
    "JAZZ / BLUES": [8],
    "JUEGOS": [9],
    "LITERATURA": [6],
    "MAGACÍN": [9],
    "MAGACÍN INFORMATIVO": [9, 10],
    "MANUALIDADES": [4],
    "METEOROLOGÍA": [10],
    "MINISERIE": [15],
    "MODA": [13],
    "MOTOR": [16],
    "MÚSICA": [8],
    "MÚSICA CLÁSICA": [8],
    "OCIO Y AFICIONES": [],
    "PREESCOLAR": [7],
    "PROGRAMA CULTURAL": [6],
    "PROGRAMA DE MÚSICA": [8],
    "PROGRAMA DEPORTIVO": [1],
    "PROGRAMA INFANTIL": [7],
    "RELAC. PERSONALES": [],
    "RELIGIÓN": [],
    "REPORTAJES ACTUALIDAD": [10],
    "RUGBY": [1],
    "SALUD Y BIENESTAR": [2],
    "SERIE": [15],
    "SERIES": [15],
    "SERIE CIENCIA FICCIÓN": [15],
    "SERIE COMEDIA": [15],
    "SERIE DE ACCIÓN": [15],
    "SERIE DE ANIMACIÓN": [15],
    "SERIE DE AVENTURAS": [15],
    "SERIE DE HUMOR": [15, 3],
    "SERIE DE SUSPENSE": [15],
    "SERIE DRAMA": [15],
    "SERIE HISTÓRICA": [15, 6],
    "SERIE INFANTIL": [15, 7],
    "SERIE JUVENIL": [15],
    "SERIE POLICIACA": [15],
    "TEATRO": [],
    "TECNOLOGÍAS": [12],
    "TELE REALIDAD": [9],
    "TELENOVELA": [9],
    "TELEVENTA": [],
    "TENIS": [1],
    "TERTULIA": [9],
    "TOROS": [],
    "TRADICIONES POPULARES": [6],
    "TURISMO": [14],
    "VARIEDADES": [9],
    "VIDEOCLIPS": [8],
    "ÓPERA": [8]
}


class MTE(ClassificationDataset, PublicURLMixin):
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

    public_urls = ("http://ottcache.dof6.com/movistarplus/webplayer.hls/OTT/epg?from={}&span=7&channel=" +
                   "&network=movistarplus").format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    def __init__(self):
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
        subscriptions = genres.map(GENRE_TO_SUBSCRIPTION)

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
