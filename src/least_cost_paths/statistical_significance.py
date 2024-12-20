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


class StatisticalSignificance:
    """Defines the StatisticalSignificance class."""

    def __init__(self, src: Path, layer_lcps: str, layer_sps: str, dst_name: str, permutations: int = 9999) -> None:
        """Initializes the StatisticalSignificance class."""

        self.dst = Path(src.parent / dst_name).with_suffix(".csv")
        self.permutations = permutations

        # Compile source data.
        # Note: Not all columns required for statistical significance tests, including geometry, hence reading via SQL.
        engine = create_engine(f"sqlite:///{src}")
        query_fields = ", ".join(f'"{field}"' for field in ("group", "source", "target", "cost", "distance"))

        logger.info(f"Compiling source data - least-cost paths: {src}, layer={layer_lcps}.")
        self.lcps = pd.read_sql(f"select {query_fields} from \"{layer_lcps}\"", con=engine)

        logger.info(f"Compiling source data - least-cost paths uniform: {src}, layer={layer_sps}.")
        self.sps = pd.read_sql(f"select {query_fields} from \"{layer_sps}\"", con=engine)

        # Configure output DataFrame.
        groups = list(set(self.lcps["group"]))
        dst_cols = ("pvalue", "r_observed", "n_permuted", "r_permuted_min", "r_permuted_max", "r_permuted_mean",
                    "r_permuted_abs_min", "r_permuted_abs_max", "r_permuted_abs_mean")
        self.dst_df = pd.DataFrame({"group": groups, **{col: [-1.0] * len(groups) for col in dst_cols}})

    def __call__(self) -> None:
        """Executes the StatisticalSignificance class."""

        self.calculate_pvalues()

        # Export results.
        self.dst_df.to_csv(self.dst, sep=",", header=True, index=False)
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
        df.sort_values(by=["source", "target"], ignore_index=True, inplace=True)

        # Normalize distance values to account for scaling differences.
        df[cost] = df[cost] / df[cost].max()

        # Generate empty matrix.
        locations_source = sorted(set(df["source"]))
        locations_target = sorted(set(df["target"]))
        matrix = np.full((len(locations_source), len(locations_target)), 0.0)

        # Generate source-target to matrix index lookups.
        lookup_source_idx = dict(zip(locations_source, range(len(locations_source))))
        lookup_target_idx = dict(zip(locations_target, range(len(locations_target))))

        # Populate matrix.
        matrix[df["source"].map(lookup_source_idx), df["target"].map(lookup_target_idx)] = df[cost].values

        return matrix

    @staticmethod
    def _gen_msr_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Generates a randomized matrix using Moran spectral randomization (MSR) via singular value decomposition (SVD).

        :param np.ndarray matrix: Cost-distance matrix.
        :return np.ndarray: Randomized cost-distance matrix.
        """

        # Decompose cost-distance matrix via singular value decomposition (SVD).
        u, s, vh = svd(matrix, full_matrices=False, compute_uv=True, overwrite_a=False)

        # Generate randomized permutation.
        s_permuted = np.random.permutation(s)

        # Reconstruct as a matrix via dot product.
        matrix_msr = np.dot(u * s_permuted, vh)

        return matrix_msr

    def calculate_pvalues(self) -> None:
        """Calculates two-tailed p-values for each group."""

        # Iterate groups.
        for group in sorted(set(self.lcps["group"])):

            logger.info(f"Group: {group}")

            # Subset DataFrame to group.
            lcps = self.lcps.loc[self.lcps["group"] == group].copy(deep=True)
            lcps.reset_index(drop=True, inplace=True)
            sps = self.sps.loc[self.sps["group"] == group].copy(deep=True)
            sps.reset_index(drop=True, inplace=True)

            # Generate cost-distance matrices.
            matrix_lcp = self._gen_cd_matrix(lcps, cost="cost")
            matrix_sp = self._gen_cd_matrix(sps, cost="distance")

            # Calculate p-value.

            # Flatten cost-distance matrices.
            distances_lcp = matrix_lcp.flatten()
            distances_sp = matrix_sp.flatten()

            # Calculate Pearson's correlation coefficient (r) based on the observed data.
            r = pearsonr(distances_lcp, distances_sp, alternative="two-sided")[0]

            # Iterate permutations.
            r_permuted = list()
            for _ in tqdm(range(self.permutations), desc=f"Generating MSR matrices and calculating Pearson's "
                                                         f"correlation coefficients (r)"):

                # Generate  MSR matrix.
                matrix_msr = self._gen_msr_matrix(matrix_lcp)

                # Flatten MSR matrix.
                distances_msr = matrix_msr.flatten()

                # Calculate Pearson's correlation coefficient (r) based on the permuted data.
                r_ = pearsonr(distances_msr, distances_sp, alternative="two-sided")[0]
                r_permuted.append(r_)

            # Calculate p-value (two-tailed).
            r_permuted = np.array(r_permuted)
            pvalue = (np.sum(np.abs(r_permuted) >= np.abs(r)) + 1) / (self.permutations + 1)

            # Store results.
            flag = self.dst_df["group"] == group
            self.dst_df.loc[flag, "pvalue"] = pvalue

            # Store results - supplementary.
            self.dst_df.loc[flag, "r_observed"] = r
            self.dst_df.loc[flag, "n_permuted"] = self.permutations
            self.dst_df.loc[flag, "r_permuted_min"] = min(r_permuted)
            self.dst_df.loc[flag, "r_permuted_max"] = max(r_permuted)
            self.dst_df.loc[flag, "r_permuted_mean"] = np.mean(r_permuted)
            self.dst_df.loc[flag, "r_permuted_abs_min"] = min(map(abs, r_permuted))
            self.dst_df.loc[flag, "r_permuted_abs_max"] = max(map(abs, r_permuted))
            self.dst_df.loc[flag, "r_permuted_abs_mean"] = np.mean(tuple(map(abs, r_permuted)))


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.argument("layer_lcps", type=click.STRING)
@click.argument("layer_sps", type=click.STRING)
@click.argument("dst_name", type=click.STRING)
@click.argument("permutations", type=click.INT, default=9999)
def main(src: Path, layer_lcps: str, layer_sps: str, dst_name: str, permutations: int = 9999) -> None:
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
    :param int permutations: Number of permutations to evaluate the null hypothesis.
    """

    try:

        statistical_significance = StatisticalSignificance(src, layer_lcps, layer_sps, dst_name, permutations)
        statistical_significance()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
