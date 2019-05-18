#
#
#   Media Tagging Engine
#
#

import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

from ..utils.io_utils import read_from_json
from .dataset import Dataset, DatasetInfo, ClassLabel


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


EPG_URL = "http://ottcache.dof6.com/movistarplus/webplayer.hls/OTT/epg?from={now}&span=1&channel=&network=movistarplus"


class MTE(Dataset):
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

    def _info(self):
        return DatasetInfo(
            features=str,
            labels=ClassLabel(names=["deportes", "salud y belleza", "humor", "hogar", "cine", "cultural y educativo",
                                     "infantil", "música", "entretenimiento", "información y actualidad", "documental",
                                     "tecnología", "moda", "viajes", "serie", "motor", "cocina"])
        )

    def _download_and_prepare(self, dl_manager):
        epg_path = dl_manager.download(EPG_URL.format(now=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))

        epg_json = read_from_json(epg_path)

        self.sheets_paths_ = []

        for channel in tqdm(epg_json, leave=False, desc="Channels"):
            for program in tqdm(channel, desc="Programs"):
                sheet_url = program.get("Ficha")
                if sheet_url is None:
                    print("Sheet url not found. Skipping")
                    continue

                try:
                    sheet_path = dl_manager.download(sheet_url, verbose=False)
                except Exception as exception:
                    print("Error downloading sheet with url {}".format(sheet_url))
                    continue

                self.sheets_paths_.append(sheet_path)


    def _load(self):
        df = pd.DataFrame(columns=["title", "description", "genre", "channel_id"])
        for idx, sheet_path in enumerate(tqdm(self.sheets_paths_)):
            sheet = read_from_json(sheet_path)
            df.loc[idx] = self.__parse_sheet(sheet)

        df.drop_duplicates(inplace=True)
        df.fillna({"title": ""}, inplace=True)
        df = df[df != ""]
        df = df[df.genre != "SIN CLASIFICAR"]
        df.dropna(subset=["genre", "description"], inplace=True)

        df["subscriptions"] = df.genre.map(GENRE_TO_SUBSCRIPTION)

        if df.subscriptions.isna().sum() > 0:
            unknown_genres = df.genre[df.subscriptions.isna()].unique().tolist()
            self.logger.warning("Unknown subscription for the following genres: {}".format(unknown_genres))

        df.dropna(subset=["subscriptions"], inplace=True)

        X = df.title + " " + df.description
        y = MultiLabelBinarizer().fit_transform(df.subscriptions)

        return X, y

    def __parse_sheet(self, sheet):
        title = sheet.get("Titulo")
        description = sheet.get("Descripcion")

        try:
            channel_id = sheet["Pases"][0]["Canal"]["CodCadenaTv"]
        except (KeyError, IndexError):
            channel_id = None

        genre = sheet.get("Genero", {}).get("ComAntena")

        return (title, description, genre, channel_id)
