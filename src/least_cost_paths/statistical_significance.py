import click
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

    def __init__(self, src: Path, layer_lcps: str, layer_lcps_uniform: str, dst_name: str) -> None:
        """Initializes the StatisticalSignificance class."""

        self.dst = Path(src.parent / dst_name).with_suffix(".csv")
        self.permutations = 999

        # Compile source data.
        # Note: Not all columns required for statistical significance tests, including geometry, hence reading via SQL.
        engine = create_engine(f"sqlite:////{src}")
        query_fields = ", ".join(f'"{field}"' for field in ("group", "source", "target", "cost", "distance"))

        logger.info(f"Compiling source data - least-cost paths: {src}, layer={layer_lcps}.")
        self.lcps = pd.read_sql(f"select {query_fields} from \"{layer_lcps}\"", con=engine)

        logger.info(f"Compiling source data - least-cost paths uniform: {src}, layer={layer_lcps_uniform}.")
        self.lcps_uniform = pd.read_sql(f"select {query_fields} from \"{layer_lcps_uniform}\"", con=engine)

        # Sort DataFrames to ensure alignment.
        self.lcps.sort_values(by=["group", "source", "target"], ignore_index=True, inplace=True)
        self.lcps_uniform.sort_values(by=["group", "source", "target"], ignore_index=True, inplace=True)

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

            # Compile dissimilarity (distance) values ('cost' for weighted LCPs, 'distance' for uniform LCPs).
            distances = self.lcps.loc[self.lcps["group"] == group, "cost"]\
                .reset_index(drop=True)
            distances_uniform = self.lcps_uniform.loc[self.lcps_uniform["group"] == group, "distance"]\
                .reset_index(drop=True)

            # Calculate Pearson's correlation coefficient based on the observed data.
            correlation_observed = distances.corr(distances_uniform, method="pearson")

            # Permute data.
            correlation_permuted = list()
            for _ in range(self.permutations):

                # Shuffle distances.
                distances_ = distances.sample(frac=1, replace=False, ignore_index=True)

                # Calculate Pearson's correlation coefficient based on the permuted data.
                correlation_permuted.append(distances_.corr(distances_uniform, method="pearson"))

            # Calculate and store p-value (two-tailed).
            pvalue = np.mean(np.abs(correlation_permuted) >= np.abs(correlation_observed))
            self.pvalues.loc[self.pvalues["group"] == group, "pvalue"] = pvalue

@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_lcps", type=click.STRING)
@click.argument("layer_lcps_uniform", type=click.STRING)
@click.argument("dst_name", type=click.STRING)
def main(src: Path, layer_lcps: str, layer_lcps_uniform: str, dst_name: str) -> None:
    """
    \b
    Description: Calculates the two-tailed p-value for each group of least-cost paths from two GeoPackage layers; one
    layer containing weighted least-cost path results and another containing uniform surface (shortest path) results.

    \b
    Assumptions:
        - Input GeoPackage layers containing least-cost paths will:
            - Have the same schema and spatial properties as those exported by least_cost_paths.py.
            - Contain results for the same set of groups and source-target point pairs.
            - Contain results for every source-target point pair permutation.
            - Contain neither duplicate source-target point pairs within any group nor reversed pairs.

    \b
    :param Path src: GeoPackage (.gpkg) containing least-cost paths layer.
    :param str layer_lcps: GeoPackage layer containing least-cost paths and their cost values.
    :param str layer_lcps_uniform: GeoPackage layer containing least-cost paths and their distance values, based on a
        uniform surface (i.e. actual shortest paths).
    :param str dst_name: Output CSV file name (excluding .csv suffix).
    """

    try:

        statistical_significance = StatisticalSignificance(src, layer_lcps, layer_lcps_uniform, dst_name)
        statistical_significance()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
