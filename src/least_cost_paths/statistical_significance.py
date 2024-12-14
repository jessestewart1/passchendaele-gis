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

    def __init__(self, src: Path, layer_lcps: str, layer_lcps_uniform: str, src_lookup: Path, layer_lookup: str,
                 dst_name: str) -> None:
        """Initializes the StatisticalSignificance class."""

        self.dst = Path(src.parent / dst_name).with_suffix(".csv")
        self.permutations = 9999

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

        # Compile index-geometry lookup data.
        logger.info(f"Compiling index-geometry lookup data: {src_lookup}, layer={layer_lookup}.")
        lookup = gpd.read_file(src_lookup, layer=layer_lookup)

        # Configure lookup dict.
        self.lookup_idx_pt = dict(zip(lookup["index"], lookup["geometry"]))
        self.crs = lookup.crs

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

        logger.info("Creating Euclidean distance matrix.")

        # Compile Point geometries for each source-target index.
        pts_source = gpd.GeoSeries(self.lcps["source"].map(self.lookup_idx_pt), crs=self.crs)
        pts_target = gpd.GeoSeries(self.lcps["target"].map(self.lookup_idx_pt), crs=self.crs)

        # Create Series of Euclidean distances for each source-target pair.
        self.lcps["euclidean"] = pts_source.distance(pts_target)

        # Iterate groups.
        for group in set(self.lcps["group"]):

            logger.info(f"Calculating p-value for group: {group}.")

            # Compile dissimilarity (distance) values.
            distances = self.lcps.loc[self.lcps["group"] == group, "cost"].reset_index(drop=True)
            distances_euclidean = self.lcps.loc[self.lcps["group"] == group, "euclidean"].reset_index(drop=True)
            distances_uniform = self.lcps_uniform.loc[self.lcps_uniform["group"] == group, "distance"]\
                .reset_index(drop=True)

            # Normalize distance values to account for scaling differences.
            distances = distances / distances.max()
            distances_euclidean = distances_euclidean / distances_euclidean.max()
            distances_uniform = distances_uniform / distances_uniform.max()

            # Calculate Pearson's correlation coefficient (r) based on the observed data.
            r_observed_ab = distances.corr(distances_uniform, method="pearson")
            r_observed_ac = distances.corr(distances_euclidean, method="pearson")
            r_observed_bc = distances_uniform.corr(distances_euclidean, method="pearson")
            r_observed = ((r_observed_ab - r_observed_ac * r_observed_bc) /
                          np.sqrt((1 - r_observed_ac**2) * (1 - r_observed_bc**2)))

            # Permute data.
            r_permuted = list()
            for _ in range(self.permutations):

                # Shuffle distances.
                distances_ = distances.sample(frac=1, replace=False, ignore_index=True)

                # Calculate Pearson's correlation coefficient (r) based on the permuted data.
                r_ab_ = distances_.corr(distances_uniform, method="pearson")
                r_ac_ = distances_.corr(distances_euclidean, method="pearson")
                r_ = (r_ab_ - r_ac_ * r_observed_bc) / np.sqrt((1 - r_ac_**2) * (1 - r_observed_bc**2))
                r_permuted.append(r_)

            # Calculate and store p-value (two-tailed).
            r_permuted = np.array(r_permuted)
            pvalue = (np.sum(np.abs(r_permuted) >= np.abs(r_observed)) + 1) / (self.permutations + 1)
            self.pvalues.loc[self.pvalues["group"] == group, "pvalue"] = pvalue

@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_lcps", type=click.STRING)
@click.argument("layer_lcps_uniform", type=click.STRING)
@click.argument("src_lookup", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True,
                                              path_type=Path))
@click.argument("layer_lookup", type=click.STRING)
@click.argument("dst_name", type=click.STRING)
def main(src: Path, layer_lcps: str, layer_lcps_uniform: str, src_lookup: Path, layer_lookup: str,
         dst_name: str) -> None:
    """
    \b
    Description: Calculates the two-tailed p-value for each group of least-cost paths from three GeoPackage layers:
        1. Weighted least-cost path results.
        2. Uniform surface (shortest path) results.
        3. Point geometries and their indexes for lookup, used to account for spatial autocorrelation.

    \b
    Assumptions:
        - Input GeoPackage layers containing least-cost paths will:
            - Have the same schema and spatial properties as those exported by least_cost_paths.py.
            - Contain results for the same set of groups and source-target point pairs.
            - Contain results for every source-target point pair permutation.
            - Contain neither duplicate source-target point pairs within any group nor reversed pairs.
            - Contain source-target values as indexes, all of which are present in the index-geometry lookup layer.
        - Input GeoPackage layer containing index-geometry lookup data will:
            - Contain field 'geometry' containing Points.
            - Contain field 'index' containing lookup indexes.
            - Contain the index for every source and target index present in the GeoPackage layers containing
              least-cost paths.

    \b
    :param Path src: GeoPackage (.gpkg) containing least-cost paths layer.
    :param str layer_lcps: GeoPackage layer containing least-cost paths and their cost values.
    :param str layer_lcps_uniform: GeoPackage layer containing least-cost paths and their distance values, based on a
        uniform surface (i.e. actual shortest paths).
    :param Path src_lookup: GeoPackage (.gpkg) containing index-geometry lookup data.
    :param str layer_lookup: GeoPackage layer containing index-geometry lookup data.
    :param str dst_name: Output CSV file name (excluding .csv suffix).
    """

    try:

        statistical_significance = StatisticalSignificance(src, layer_lcps, layer_lcps_uniform, src_lookup,
                                                           layer_lookup, dst_name)
        statistical_significance()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
