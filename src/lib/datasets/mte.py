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
from .dataset import Dataset, DatasetInfo, ClassLabel, Tensor


GENRE_TO_SUBSCRIPTION = {
    'ATLETISMO': [0],
    'BALONCESTO': [0],
    'BALONMANO': [0],
    'BELLEZA': [1],
    'BRICOLAJE': [3],
    'CAZA Y PESCA': [0],
    'CICLISMO': [0],
    'CINE': [4],    'CINE ADULTO': [4],
    'CINE CIENCIA FICCIÓN': [4],
    'CINE COMEDIA': [4, 2],
    'CINE COMEDIA ROMANT.': [4, 2],
    'CINE DE ACCIÓN': [4],
    'CINE DE ANIMACIÓN': [4],
    'CINE DE AVENTURAS': [4],
    'CINE DRAMA': [4],
    'CINE HISTÓRICO': [4, 5],
    'CINE INFANTIL': [4, 6],
    'CINE MUSICAL': [4, 7],
    'CINE OESTE': [4],
    'CINE POLICIACO': [4],
    'CINE SUSPENSE': [4],
    'CINE TERROR': [4],
    'CINE TV': [4],
    'COCINA': [16],
    'CONCIERTO': [7],
    'CONCURSO': [8],
    'CORAZÓN / SOCIEDAD': [8],
    'CORTO': [],    'CULTURAL/EDUCATIVO': [5],
    'DANCE / ELECTRÓNICA': [7],
    'DANZA / BALLET': [7],
    'DEBATE': [9],
    'DECORACIÓN': [3],
    'DEPORTE': [0],
    'DEPORTES': [0],
    'DEPORTE ACUÁTICO': [0],
    'DEPORTE DE INVIERNO': [0],
    'DEPORTE TRADICIONAL': [0],
    'DIBUJOS ANIMADOS': [6],
    'DOC. ACTUALIDAD': [10],
    'DOC. ARTE Y CULTURA': [10, 5],
    'DOC. CAZA Y PESCA': [10, 0],
    'DOC. CIENCIA Y TEC.': [10, 11],
    'DOC. NATURALEZA': [10],
    'DOC. TENDENCIAS': [10, 12],
    'DOC. VIAJES': [10, 13],
    'DOCU-SERIE': [10, 14],
    'DOCUMENTAL': [10],
    'DOCUMENTAL BIOGRAFÍA': [10],
    'DOCUMENTAL DE CINE': [10, 4],
    'DOCUMENTAL HISTORIA': [10, 5],
    'DOCUMENTAL MÚSICA': [10, 7],
    'DOCUMENTAL POLÍTICA': [10, 9],
    'DOCUMENTAL SOCIEDAD': [10],
    'ECONOMÍA': [9],
    'ENTRETENIM. DEBATE': [8],
    'ENTRETENIM. ENTREVISTA': [8],
    'ENTRETENIMIENTO': [8],
    'ENTREVISTA': [8, 9],
    'ESOTERISMO': [],
    'FORMACIÓN ACADÉMICA': [5],
    'FÚTBOL': [0],
    'FÚTBOL AMERICANO': [0],
    'GOLF': [0],    'HOCKEY': [0],
    'HUMOR': [2],
    'IDIOMAS': [5],
    'INF. SOCIEDAD': [],
    'INFANTIL': [6],
    'INFANTIL EDUCATIVO': [6],
    'INFORMACIÓN': [9],
    'INFORMACIÓN DEPORTE': [9, 0],
    'INFORMACIÓN POLÍTICA': [9],
    'INFORMATIVO': [9],
    'JAZZ / BLUES': [7],
    'JUEGOS': [8],
    'LITERATURA': [5],
    'MAGACÍN': [8],
    'MAGACÍN INFORMATIVO': [8, 9],
    'MANUALIDADES': [3],
    'METEOROLOGÍA': [9],
    'MINISERIE': [14],
    'MODA': [12],
    'MOTOR': [15],
    'MÚSICA': [7],
    'MÚSICA CLÁSICA': [7],
    'OCIO Y AFICIONES': [],
    'PREESCOLAR': [6],
    'PROGRAMA CULTURAL': [5],
    'PROGRAMA DE MÚSICA': [7],
    'PROGRAMA DEPORTIVO': [0],
    'PROGRAMA INFANTIL': [6],
    'RELAC. PERSONALES': [],
    'RELIGIÓN': [],
    'REPORTAJES ACTUALIDAD': [9],
    'RUGBY': [0],
    'SALUD Y BIENESTAR': [1],
    'SERIE': [14],
    'SERIES': [14],
    'SERIE CIENCIA FICCIÓN': [14],
    'SERIE COMEDIA': [14],
    'SERIE DE ACCIÓN': [14],
    'SERIE DE ANIMACIÓN': [14],
    'SERIE DE AVENTURAS': [14],
    'SERIE DE HUMOR': [14, 2],
    'SERIE DE SUSPENSE': [14],
    'SERIE DRAMA': [14],
    'SERIE HISTÓRICA': [14, 5],
    'SERIE INFANTIL': [14, 6],
    'SERIE JUVENIL': [14],
    'SERIE POLICIACA': [14],
    'TEATRO': [],
    'TECNOLOGÍAS': [11],
    'TELE REALIDAD': [8],
    'TELENOVELA': [8],
    'TELEVENTA': [],
    'TENIS': [0],
    'TERTULIA': [8],
    'TOROS': [],    'TRADICIONES POPULARES': [5],
    'TURISMO': [13],
    'VARIEDADES': [8],
    'VIDEOCLIPS': [7],
    'ÓPERA': [7]
}

EPG_URL = "http://ottcache.dof6.com/movistarplus/webplayer.hls/OTT/epg?from={now}&span=7&channel=&network=movistarplus"

CLASS_NAMES = ["Deportes", "Salud y Belleza", "Humor", "Hogar", "Cine", "Cultural y Educativo", "Infantil", "Música",
               "Entretenimiento", "Información y Actualidad", "Documental", "Tecnología", "Moda", "Viajes", "Serie",
               "Motor", "Cocina"]


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

    def info(self):
        return DatasetInfo(
            features=Tensor(shape=[], dtype=str),
            labels=ClassLabel(names=CLASS_NAMES, multilabel=True)
        )

    def download_and_prepare(self, dl_manager):
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

        df = pd.DataFrame(columns=["title", "description", "genre", "channel_id"])
        for idx, sheet_path in enumerate(tqdm(self.sheets_paths_)):
            sheet = read_from_json(sheet_path)
            df.loc[idx] = self.__parse_sheet(sheet)

        df.drop_duplicates(inplace=True)
        df.fillna({"title": ""}, inplace=True)
        df = df[df != ""]  # convert empty strings to NaN
        df = df[df.genre != "SIN CLASIFICAR"]
        df.dropna(subset=["genre", "description"], inplace=True)

        df["subscriptions"] = df.genre.map(GENRE_TO_SUBSCRIPTION)

        df.dropna(subset=["subscriptions"], inplace=True)

        self.features_ = df.title + " " + df.description
        self.labels_ = MultiLabelBinarizer(classes=range(len(CLASS_NAMES))).fit_transform(df.subscriptions)


    def __parse_sheet(self, sheet):
        title = sheet.get("Titulo")
        description = sheet.get("Descripcion")

        try:
            channel_id = sheet["Pases"][0]["Canal"]["CodCadenaTv"]
        except (KeyError, IndexError):
            channel_id = None

        genre = sheet.get("Genero", {}).get("ComAntena")

        return (title, description, genre, channel_id)

    def __getitem__(self, index):
        return self.features_.iloc[index], self.labels_[index]

    def __len__(self):
        return len(self.features_)
