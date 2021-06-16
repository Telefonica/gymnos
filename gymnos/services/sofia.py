#
#
#   SOFIA
#
#

import re
import os
import fastdl
import requests

from dataclasses import dataclass
from posixpath import join as urljoin

from ..config import get_gymnos_config, get_gymnos_home


Response = requests.models.Response


class NotLoggedIn(Exception):
    """
    Raises when not logged in to SOFIA
    """

    def __init__(self):
        message = ("This functionality requires to be logged."
                   "Please run gymnos-login to log in.")
        super().__init__(message)


@dataclass
class SOFIADataset:
    username: str
    name: str

    @classmethod
    def parse(cls, dataset):
        match = re.match(r"^(.+)/datasets/(.+)$", dataset)

        if not match:
            raise ValueError("Unexpected dataset {}. It must be in the following format <username>/datasets/<name>")

        username, dataset_name = match.group(1), match.group(2)

        return cls(username, dataset_name)


def login_required(func):
    config = get_gymnos_config()
    if config.sofia.access_token is None:
        raise NotLoggedIn()

    return func


class SOFIA:

    domain = os.getenv("SOFIA_DOMAIN", "http://localhost:5555")  # FIXME

    @classmethod
    def session(cls):
        config = get_gymnos_config()
        if config.sofia.access_token is None:
            raise NotLoggedIn()

        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {config.sofia.access_token}"})
        return session

    @classmethod
    def login(cls, username_or_email: str, password: str) -> Response:
        return requests.post(urljoin(cls.domain, "api", "auth", "login"), json={
            "username_or_email": username_or_email,
            "password": password
        })

    @classmethod
    def get_current_user(cls):
        return cls.session().get(urljoin(cls.domain, "api", "user"))

    @classmethod
    def get_dataset_files(cls, dataset: str):
        dataset = SOFIADataset.parse(dataset)
        return cls.session().get(urljoin(cls.domain, "api", "datasets", dataset.username, dataset.name,
                                         "files"))

    @classmethod
    def create_project_job(cls, args, project_name, ref=None, device="CPU", name=None, description=None):
        response = cls.get_current_user()
        response.raise_for_status()
        username = response.json()["username"]

        json_data = {
            "args": args,
            "ref": ref,
            "device": device,
            "description": description
        }

        if name is not None:
            json_data["name"] = name  # null is not allowed

        return cls.session().post(urljoin(cls.domain, "api", "projects", username, project_name, "jobs"),
                                  json=json_data)

    @classmethod
    def download_dataset(cls, dataset, files=None, force_download=False, max_workers=None):
        home = get_gymnos_home()

        if files is None:
            response = cls.get_dataset_files(dataset)
            response.raise_for_status()
            files = response.json()

        dataset = SOFIADataset.parse(dataset)

        config = get_gymnos_config()

        save_dir = os.path.join(home, "datasets", "sofia", dataset.username, dataset.name)

        with fastdl.Parallel(max_workers=max_workers) as p:
            downloads = []

            for file in files:
                download = p.download(
                    url=urljoin(cls.domain, "api", "datasets", dataset.username, dataset.name, "files", file["name"],
                                "download"),
                    headers={
                        "Authorization": f"Bearer {config.sofia.access_token}"
                    },
                    content_disposition=True,
                    dir_prefix=save_dir,
                    force_download=force_download
                )
                downloads.append(download)

            for download in downloads:
                download.get()

        return save_dir

    @classmethod
    @login_required
    def download_model(cls, username, model):
        ...
