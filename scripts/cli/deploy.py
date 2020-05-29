#
#
#   Deploy CLI app
#
#

import os

from ..utils import ask
from ..utils.io_utils import read_json

from pprint import pprint
from gymnos import config
from gymnos.services.sofia import DEFAULT_DOMAIN
from gymnos.utils.lazy_imports import lazy_imports as lazy


class Config(config.Config):
    SOFIA_EMAIL = config.Value(required=True, help="SOFIA account email")
    SOFIA_PASSWORD = config.Value(required=True, help="SOFIA account password")
    SOFIA_DOMAIN = config.Value(required=False, help="SOFIA domain", default=DEFAULT_DOMAIN)


def add_arguments(parser):
    parser.add_argument("saved_trainer", help="Saved trainer file path", type=str)
    parser.add_argument("--metadata", help=("JSON file containing metadata. The required keys are the following: title."
                                            "The optional keys are the following: description, license, "
                                            "acknowledgements, public"),
                        type=str, required=False)


def run_command(args):
    if not os.path.isfile(args.saved_trainer):
        raise FileNotFoundError(args.saved_trainer)

    config = Config()
    config.load()

    session = lazy.requests.Session()
    session.hooks = {
        "response": lambda r, *args, **kwargs: r.raise_for_status()
    }

    res = session.post(config.SOFIA_DOMAIN + "/api/login", data=dict(
        email=config.SOFIA_EMAIL,
        password=config.SOFIA_PASSWORD
    ))

    auth_headers = {
        "Authorization": "Bearer " + res.json()["token"]
    }

    if args.metadata:
        metadata = read_json(args.metadata)
    else:
        metadata = dict(
            title=ask.text("Title", required=True),
            description=ask.text("Description"),
            public=ask.confirm("Is it public?", default=True)
        )

    with open(args.saved_trainer, "rb") as fp:
        res = session.post(config.SOFIA_DOMAIN + "/api/models", data=metadata, files=dict(model=fp),
                           headers=auth_headers)

    print("Saved trainer uploaded successfully")

    pprint(res.json())
