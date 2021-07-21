#
#
#   Save login for SOFIA
#
#

import click

from rich import print as rprint

from ..services.sofia import SOFIA
from ..config import get_gymnos_config, set_gymnos_config


@click.command(help="Login to SOFIA")
@click.option("--username_or_email", prompt=True, help="SOFIA username or email")
@click.option("--password", prompt=True, hide_input=True, help="SOFIA password")
def main(username_or_email, password):
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
