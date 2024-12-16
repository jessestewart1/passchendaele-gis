import click
import logging
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.linalg import svd
from scipy.stats import pearsonr
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

        # Configure output DataFrame.
        groups = list(set(self.lcps["group"]))
        self.pvalues = pd.DataFrame({"group": groups, "pvalue": [-1] * len(groups)})

    def __call__(self) -> None:
        """Executes the StatisticalSignificance class."""

        self.calculate_pvalues()

        # Export results.
        self.pvalues.to_csv(self.dst, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst}.")

    @staticmethod
    def _gen_cd_matrix(df: pd.DataFrame, cost: str) -> np.ndarray:
        """
        Generates a cost-distance matrix from a DataFrame.

        :param pd.DataFrame df: DataFrame containing source locations, target locations, and cost values.
        :param str cost: Field name containing cost values.
        :return np.ndarray: Cost-distance matrix.
        """

        logger.info(f"Generating cost-distance matrix using cost field: {cost}.")

        # Sort DataFrame to ensure alignment.
        df.sort_values(by=["group", "source", "target"], ignore_index=True, inplace=True)

        # Normalize distance values to account for scaling differences.
        df[cost] = df[cost] / df[cost].max()

        # Generate empty matrix.
        locations_source = sorted(set(df["source"]))
        locations_target = sorted(set(df["target"]))
        matrix = np.full((len(locations_source), len(locations_target)), 0.0)

        # Iteratively populate matrix.
        for i, source in enumerate(locations_source):
            for j, target in enumerate(locations_target):
                matrix[i, j] = df.loc[(df["source"] == source) & (df["target"] == target), cost]

        return matrix

    def _gen_msr_matrices(self, matrix: np.ndarray) -> [np.ndarray, ...]:
        """
        Generates Moran spectral randomization (MSR) matrices using singular value decomposition (SVD).

        :param np.ndarray matrix: Cost-distance matrix.
        :return [np.ndarray, ...]: List of randomized cost-distance matrices.
        """

        logger.info(f"Generating {self.permutations} Moran spectral randomization (MSR) matrices.")

        # Decompose cost-distance matrix via singular value decomposition (SVD).
        u, s, vh = svd(matrix, full_matrices=False, compute_uv=True, overwrite_a=False)

        # Generate randomized matrices.
        matrices_msr = list()
        for _ in range(self.permutations):

            # Generate randomized permutation.
            s_permuted = np.random.permutation(s)

            # Reconstruct as a matrix via dot product.
            matrix_msr = np.dot(u * s_permuted, vh)
            matrices_msr.append(matrix_msr)

        return matrices_msr

    def calculate_pvalues(self) -> None:
        """Calculates two-tailed p-values for each group."""

        # Iterate groups.
        for group in set(self.lcps["group"]):

            # Generate cost-distance matrices.
            flag_group = self.lcps["group"] == group
            matrix_lcp = self._gen_cd_matrix(self.lcps.loc[flag_group], cost="cost")
            matrix_sp = self._gen_cd_matrix(self.sps.loc[flag_group], cost="distance")

            # Generate Moran spectral randomization (MSR) matrices.
            matrices_msr = self._gen_msr_matrices(matrix_lcp)

            logger.info(f"Calculating p-value for group: {group}.")

            # Flatten cost-distance matrices.
            distances_lcp = matrix_lcp.flatten()
            distances_sp = matrix_sp.flatten()

            # Calculate Pearson's correlation coefficient (r) based on the observed data.
            r = pearsonr(distances_lcp, distances_sp, alternative="two-sided")[0]

            # Permute data.
            r_permuted = list()
            for matrix_msr in matrices_msr:

                # Flatten MSR matrix.
                distances_msr = matrix_msr.flatten()

                # Calculate Pearson's correlation coefficient (r) based on the permuted data.
                r_ = pearsonr(distances_msr, distances_sp, alternative="two-sided")[0]
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
