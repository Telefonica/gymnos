#
#
#   SOFIA
#
#

import re
import os
import json
import fastdl
import requests

from omegaconf import OmegaConf
from dataclasses import dataclass
from posixpath import join as urljoin

from ..config import get_gymnos_config, get_gymnos_home


Response = requests.models.Response


class NotLoggedIn(Exception):
    """
    Raises when not logged in to SOFIA
    """

    def __init__(self):
        message = ("This functionality requires to be logged. "
                   "Please run gymnos-login to log in.")
        super().__init__(message)


@dataclass
class SOFIADataset:
    username: str
    name: str

    @classmethod
    def parse(cls, dataset):
        username, name = parse_resource("datasets", dataset)
        return cls(username, name)


@dataclass
class SOFIAModel:
    username: str
    name: str

    @classmethod
    def parse(cls, model):
        username, name = parse_resource("models", model)
        return cls(username, name)


def parse_resource(resource_type: str, resource: str):
    match = re.match(rf"^(.+)/{resource_type}/(.+)$", resource)

    if not match:
        raise ValueError("Unexpected dataset {}. It must be in the following format <username>/datasets/<name>")

    username, name = match.group(1), match.group(2)

    return username, name


def login_required(func):
    config = get_gymnos_config()
    if config.sofia.access_token is None:
        raise NotLoggedIn()

    return func


class SOFIA:
    """
    Service to interact with SOFIA API.

    Downloads will be stored on GYMNOS_HOME/downloads/sofia/
    """

    domain = os.getenv("SOFIA_DOMAIN", "https://sofia.eu.ngrok.io")

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
    def get_model(cls, model: str):
        model = SOFIAModel.parse(model)
        return cls.session().get(urljoin(cls.domain, "api", "models", model.username, model.name))

    @classmethod
    def download_model_artifacts(cls, model: str, force_download=False, force_extraction=False, verbose=True):
        _ = cls.session()  # check credentials

        model = SOFIAModel.parse(model)
        config = get_gymnos_config()
        home = get_gymnos_home()
        save_dir = os.path.join(home, "downloads", "sofia", "models", model.username, model.name)
        download_url = urljoin(cls.domain, "api", "models", model.username, model.name, "artifacts", "download")

        fastdl.download(
            url=download_url,
            headers={
                "Authorization": f"Bearer {config.sofia.access_token}"
            },
            progressbar=verbose,
            fname="artifacts.zip",
            dir_prefix=save_dir,
            extract=True,
            force_download=force_download,
            force_extraction=force_extraction
        )

        return os.path.join(save_dir, "artifacts")

    @classmethod
    def create_project_job(cls, args, project_name, ref=None, device="CPU", name=None, description=None,
                           notify_on_completion=False):
        response = cls.get_current_user()
        response.raise_for_status()

        username = response.json()["username"]

        json_data = {
            "args": args,
            "ref": ref,
            "device": device,
            "description": description,
            "notify_on_completion": notify_on_completion
        }

        if name is not None:
            json_data["name"] = name  # null is not allowed

        return cls.session().post(urljoin(cls.domain, "api", "projects", username, project_name, "jobs"),
                                  json=json_data)

    @classmethod
    def get_project_job(cls, username, project_name, job_name):
        return cls.session().get(urljoin(cls.domain, "api", "projects", username, project_name, "jobs", job_name))

    @classmethod
    def get_project_job_logs(cls, username, project_name, job_name, lineno: int = 0):
        return cls.session().get(urljoin(cls.domain, "api", "projects", username, project_name, "jobs", job_name,
                                         "task_logs"), params={"lineno": lineno})

    @classmethod
    def create_model(cls, name, description, is_public, module, predictors, config, run_info, artifacts_path):
        model = {
            "name": name,
            "description": description,
            "is_public": is_public,
            "gymnos_module": module,
            "gymnos_predictors": predictors,
            "run": run_info
        }

        with open(artifacts_path, "rb") as fp:
            files = [
                ("model", ("model", json.dumps(model), "application/json")),
                ("config", ("config.yaml", OmegaConf.to_yaml(config), "application/x-yaml")),
                ("artifacts", ("artifacts.zip", fp, "application/zip"))
            ]

            return cls.session().post(urljoin(cls.domain, "api", "user", "models"), files=files)

    @classmethod
    def download_dataset(cls, dataset, files=None, force_download=False, max_workers=None) -> str:
        """
        Download dataset from SOFIA platform.
        Files will be downloaded in parallel

        Parameters
        ----------
        dataset
            Dataset to download (`<username>/datasets/<dataset>`), e.g ``johndoe/datasets/mydataset``
        files
            Files to download. By default, all files are downloaded
        force_download
            Whether or not ignore cache to download files
        max_workers
            Max workers for parallel downloads. Defaults to number of CPUs.

        Examples
        ----------
        >>> download_dir = SOFIA.download_dataset("johndoe/datasets/super-dataset")

        >>> download_dir = SOFIA.download_dataset("janedoe/datasets/custom-dataset", files=["data.csv", "names.json"])

        Returns
        -------
        str
            Download directory
        """
        if files is None:
            response = cls.get_dataset_files(dataset)
            response.raise_for_status()

            files = [file["name"] for file in response.json()]
        else:
            _ = cls.session()  # check credentials are stored

        home = get_gymnos_home()

        dataset = SOFIADataset.parse(dataset)

        config = get_gymnos_config()

        save_dir = os.path.join(home, "downloads", "sofia", "datasets", dataset.username, dataset.name)

        with fastdl.Parallel(max_workers=max_workers) as p:
            downloads = []

            for file in files:
                download = p.download(
                    url=urljoin(cls.domain, "api", "datasets", dataset.username, dataset.name, "files", file,
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
