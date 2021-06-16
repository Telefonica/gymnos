#
#
#   Login to SOFIA
#
#

import argparse

from getpass import getpass
from rich import print as rprint

from ..services.sofia import SOFIA
from ..config import get_gymnos_config, set_gymnos_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=False)
    parser.add_argument("--username", required=False)
    parser.add_argument("--password", required=False)

    args = parser.parse_args()

    if args.email is not None:
        username_or_email = args.email
    elif args.username is not None:
        username_or_email = args.username
    else:
        username_or_email = input("Username or email: ")

    password = args.password
    if password is None:
        password = getpass()

    config = get_gymnos_config()

    response = SOFIA.login(username_or_email, password)

    if response.status_code == 401:
        rprint(":locked:[bold red] Unauthorized. Please check credentials")
        raise SystemExit(1)

    response.raise_for_status()

    data = response.json()

    rprint(f":unlocked:[green] Successfully logged as {data['user']['full_name']}")

    config.sofia.access_token = data["access_token"]

    set_gymnos_config(config)
