import click
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import create_engine

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class StatisticalSignificance:
    """Defines the StatisticalSignificance class."""

    def __init__(self, src: Path, layer_lcps: str, layer_sps: str, dst_name: str) -> None:
        """Initializes the StatisticalSignificance class."""

        self.dst = Path(src.parent / dst_name).with_suffix(".csv")
        self.permutations = 9999

        # Compile source data.
        # Note: Not all columns required for statistical significance tests, including geometry, hence reading via SQL.
        engine = create_engine(f"sqlite:////{src}")
        query_fields = ", ".join(f'"{field}"' for field in ("group", "source", "target", "cost", "distance"))

        logger.info(f"Compiling source data - least-cost paths: {src}, layer={layer_lcps}.")
        self.lcps = pd.read_sql(f"select {query_fields} from \"{layer_lcps}\"", con=engine)

        logger.info(f"Compiling source data - least-cost paths uniform: {src}, layer={layer_sps}.")
        self.sps = pd.read_sql(f"select {query_fields} from \"{layer_sps}\"", con=engine)

        # Sort DataFrames to ensure alignment.
        self.lcps.sort_values(by=["group", "source", "target"], ignore_index=True, inplace=True)
        self.sps.sort_values(by=["group", "source", "target"], ignore_index=True, inplace=True)

        # Configure output DataFrame.
        groups = list(set(self.lcps["group"]))
        self.pvalues = pd.DataFrame({"group": groups, "pvalue": [-1] * len(groups)})

    def __call__(self) -> None:
        """Executes the StatisticalSignificance class."""

        self.calculate_pvalues()

        # Export results.
        self.pvalues.to_csv(self.dst, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst}.")

    def calculate_pvalues(self) -> None:
        """Calculates two-tailed p-values for each group."""

        # Iterate groups.
        for group in set(self.lcps["group"]):

            logger.info(f"Calculating p-value for group: {group}.")

            # Compile dissimilarity (distance) values.
            flag_group = self.lcps["group"] == group
            distances_lcp = self.lcps.loc[flag_group, "cost"].reset_index(drop=True)
            distances_sp = self.sps.loc[flag_group, "distance"].reset_index(drop=True)

            # Normalize distance values to account for scaling differences.
            distances_lcp = distances_lcp / distances_lcp.max()
            distances_sp = distances_sp / distances_sp.max()

            # Calculate Pearson's correlation coefficient (r) based on the observed data.
            r = distances_lcp.corr(distances_sp, method="pearson")

            # Permute data.
            r_permuted = list()
            for _ in range(self.permutations):

                # Shuffle distances.
                distances_lcp_ = distances_lcp.sample(frac=1, replace=False, ignore_index=True)

                # Calculate Pearson's correlation coefficient (r) based on the permuted data.
                r_ = distances_lcp_.corr(distances_sp, method="pearson")
                r_permuted.append(r_)

            # Calculate and store p-value (two-tailed).
            r_permuted = np.array(r_permuted)
            pvalue = (np.sum(np.abs(r_permuted) >= np.abs(r)) + 1) / (self.permutations + 1)
            self.pvalues.loc[self.pvalues["group"] == group, "pvalue"] = pvalue

@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_lcps", type=click.STRING)
@click.argument("layer_sps", type=click.STRING)
@click.argument("dst_name", type=click.STRING)
def main(src: Path, layer_lcps: str, layer_sps: str, dst_name: str) -> None:
    """
    \b
    Description: Calculates the two-tailed p-value for each group of least-cost paths from two GeoPackage layers:
        1. Least-cost path results.
        2. Shortest-path results.

    \b
    Assumptions:
        - Input GeoPackage layers will:
            - Have the same schema and spatial properties as those exported by least_cost_paths.py.
            - Contain results for the same set of groups and source-target point pairs.
            - Contain results for every source-target point pair permutation (i.e. no missing pairs per group).
            - Contain neither duplicate source-target point pairs within any group nor reversed pairs.

    \b
    :param Path src: GeoPackage (.gpkg) containing least-cost paths layer.
    :param str layer_lcps: GeoPackage layer containing least-cost paths and their cost values.
    :param str layer_sps: GeoPackage layer containing shortest-paths and their distance values.
    :param str dst_name: Output CSV file name (excluding .csv suffix).
    """

    try:

        statistical_significance = StatisticalSignificance(src, layer_lcps, layer_sps, dst_name)
        statistical_significance()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
