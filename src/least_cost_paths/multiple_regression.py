import click
import logging
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.linalg import svd
from scipy.stats import pearsonr
from sqlalchemy import create_engine
from tqdm import tqdm

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class MultipleRegression:
    """Defines the MultipleRegression class."""

    def __init__(self, src: Path, pvalue_slope: Path, pvalue_ground_conditions: Path, pvalue_avenues_of_approach: Path,
                 pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path, dst_name: Path) -> None:
        """Initializes the MultipleRegression class."""

        self.dst = Path(src.parent / dst_name).with_suffix(".csv")
        self.lcps = dict()
        self.pvalues = dict()
        self.alpha = 0.05

        # Compile least-cost paths.
        # Note: Not all columns required for statistical significance tests, including geometry, hence reading via SQL.
        engine = create_engine(f"sqlite:///{src}")
        query_fields = ", ".join(f'"{field}"' for field in ("group", "source", "target", "cost"))
        for layer in ("slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds", "machine_gun_viewsheds",
                      "hazard_exposure", "terrain_passability", "manoeuvrability"):

            logger.info(f"Compiling least-cost paths: {src}, layer={layer}.")
            self.lcps[layer] = pd.read_sql(f"select {query_fields} from \"{layer}\"", con=engine).copy(deep=True)

        # Compile pvalues.
        for indicator, src in {"slope": pvalue_slope,
                               "ground_conditions": pvalue_ground_conditions,
                               "avenues_of_approach": pvalue_avenues_of_approach,
                               "rifle_viewsheds": pvalue_rifle_viewsheds,
                               "machine_gun_viewsheds": pvalue_machine_gun_viewsheds}.items():

            logger.info(f"Compiling pvalues: {src} (indicator={indicator}).")
            self.pvalues[indicator] = pd.read_csv(src, sep=",", header=0).copy(deep=True)

    def __call__(self) -> None:
        """Executes the MultipleRegression class."""

        ...


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("pvalue_slope",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("pvalue_ground_conditions",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("pvalue_avenues_of_approach",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("pvalue_rifle_viewsheds",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("pvalue_machine_gun_viewsheds",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("dst_name",
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
def main(src: Path, pvalue_slope: Path, pvalue_ground_conditions: Path, pvalue_avenues_of_approach: Path,
         pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path, dst_name: Path) -> None:
    """
    \b
    Description: For each group and aggregated manoeuvrability raster, using the statistically significant
    manoeuvrability indicators specific to each group, creates a multiple linear regression equation from least-cost
    paths cost values whereby:
        - independent variable(s): manoeuvrability indicator(s).
        - dependent variable: aggregated manoeuvrability indicator.

    \b
    Output: Outputs a .csv, based on a given name, within the same directory the source pvalue CSVs, containing the
    following attributes:
        - group: Group name.
        - dependent: Aggregated manoeuvrability raster name (dependent variable).
        - equation: Regression equation.

    \b
    Assumptions:
        - Input GeoPackage layers will have the same schema and spatial properties as those exported by
          least_cost_paths.py.
        - Input GeoPackage will contain a different layer for each manoeuvrability indicator and aggregated
          manoeuvrability indicator, containing least-cost path results and named according to the indicator name:
            - slope
            - ground_conditions
            - avenues_of_approach
            - rifle_viewsheds
            - machine_gun_viewsheds.
            - hazard_exposure
            - terrain_passability
            - manoeuvrability
        - Input pvalue CSVs will contain the same schema as those exported by statistical_significance.py.

    \b
    :param Path src: GeoPackage (.gpkg) containing least-cost paths layers.
    :param Path pvalue_slope: CSV (.csv) containing pvalues for indicator=slope.
    :param Path pvalue_ground_conditions: CSV (.csv) containing pvalues for indicator=ground_conditions.
    :param Path pvalue_avenues_of_approach: CSV (.csv) containing pvalues for indicator=avenues_of_approach.
    :param Path pvalue_rifle_viewsheds: CSV (.csv) containing pvalues for indicator=rifle_viewsheds.
    :param Path pvalue_machine_gun_viewsheds: CSV (.csv) containing pvalues for indicator=rifle_viewsheds.
    :param str dst_name: Output CSV file name (excluding .csv suffix).
    """

    try:

        multiple_regression = MultipleRegression(src, pvalue_slope, pvalue_ground_conditions,
                                                 pvalue_avenues_of_approach, pvalue_rifle_viewsheds,
                                                 pvalue_machine_gun_viewsheds, dst_name)
        multiple_regression()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
