import click
import geopandas as gpd
import logging
import pandas as pd
import rasterio as rio
import sys
from pathlib import Path

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class SomeClass:
    """Defines the SomeClass class."""

    def __init__(self, ...) -> None:
        """Initializes the SomeClass class."""

        # TODO

    def __call__(self) -> None:
        """Executes the SomeClass class."""

        # TODO


@click.command()
@click.argument("example1", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument("example2", type=click.STRING)
def main(...) -> None:
    """
    \b
    Description: TODO.

    \b
    Assumptions:
        - TODO

    \b
    :param str example1: TODO.
    :param str example2: TODO.
    """

    try:

        some_class = SomeClass(...)
        some_class()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
