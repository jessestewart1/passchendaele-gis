import click
import logging
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from sqlalchemy import create_engine
from statsmodels.api import OLS
from statsmodels.stats.anova import anova_lm
from statsmodels.tools import add_constant
from tqdm import tqdm

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class RegressionAnova:
    """Defines the RegressionAnova class."""

    def __init__(self, src: Path, pvalue_slope: Path, pvalue_ground_conditions: Path, pvalue_avenues_of_approach: Path,
                 pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path) -> None:
        """Initializes the RegressionAnova class."""

        self.dst_df_equations_equal = pd.DataFrame()
        self.dst_df_equations_optimized = pd.DataFrame()
        self.dst_equations_equal = Path(src.parent / "equations_equally_weighted.csv")
        self.dst_equations_optimized = Path(src.parent / "equations_regression_optimized.csv")
        self.dst_df_anova_equal = pd.DataFrame()
        self.dst_df_anova_optimized = pd.DataFrame()
        self.dst_df_anova_between = pd.DataFrame()
        self.dst_anova_equal = Path(src.parent / "anova_equally_weighted.csv")
        self.dst_anova_optimized = Path(src.parent / "anova_regression_optimized.csv")
        self.dst_anova_between = Path(src.parent / "anova_between.csv")
        self.models_equal = defaultdict(dict)
        self.models_optimized = defaultdict(dict)
        self.lcps = dict()
        self.pvalues = dict()
        self.alpha = 0.05
        self.dependencies = {
            "hazard_exposure": ["rifle_viewsheds", "machine_gun_viewsheds"],
            "terrain_passability": ["slope", "ground_conditions", "avenues_of_approach"],
            "manoeuvrability": ["slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds",
                                "machine_gun_viewsheds"]
        }

        # Compile least-cost paths.
        # Note: Not all columns required for statistical significance tests, including geometry, hence reading via SQL.
        engine = create_engine(f"sqlite:///{src}")
        query_fields = ", ".join(f'"{field}"' for field in ("group", "source", "target", "cost"))
        for layer in ("slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds", "machine_gun_viewsheds",
                      "hazard_exposure", "terrain_passability", "manoeuvrability"):

            logger.info(f"Compiling least-cost path attribution: {src}, layer={layer}.")

            # Load DataFrame.
            df = pd.read_sql(f"select {query_fields} from \"{layer}\"", con=engine)

            # Sort values.
            self.lcps[layer] = df.sort_values(by=["group", "source", "target"], ignore_index=True).copy(deep=True)

        # Compile pvalues.
        for indicator, src in {"slope": pvalue_slope,
                               "ground_conditions": pvalue_ground_conditions,
                               "avenues_of_approach": pvalue_avenues_of_approach,
                               "rifle_viewsheds": pvalue_rifle_viewsheds,
                               "machine_gun_viewsheds": pvalue_machine_gun_viewsheds}.items():

            logger.info(f"Identifying groups with statistically significant pvalues: {src} (indicator={indicator}).")
            df = pd.read_csv(src, sep=",", header=0, usecols=["group", "pvalue"])
            self.pvalues[indicator] = set(df.loc[df["pvalue"] <= self.alpha, "group"])

    def __call__(self) -> None:
        """Executes the RegressionAnova class."""

        # Create equations.
        self.dst_df_equations_equal = self.gen_equations(equal_coeff=True)
        self.dst_df_equations_optimized = self.gen_equations(equal_coeff=False)

        # Perform and compile results of ANOVA.
        self.anova()

        # Export results.
        self.dst_df_equations_equal.to_csv(self.dst_equations_equal, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_equations_equal}.")

        self.dst_df_equations_optimized.to_csv(self.dst_equations_optimized, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_equations_optimized}.")

        self.dst_df_anova_equal.to_csv(self.dst_anova_equal, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_anova_equal}.")

        self.dst_df_anova_optimized.to_csv(self.dst_anova_optimized, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_anova_optimized}.")

        self.dst_df_anova_between.to_csv(self.dst_anova_between, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_anova_between}.")

    def anova(self) -> None:
        """
        Performs ANOVA for each group and aggregated manoeuvrability indicator for each of the following:
            1. ANOVA between each group combination using equally weighted models.
            2. ANOVA between each group combination using regression-optimized models.
            3. ANOVA between the equally weighted and regression-optimized models for each group.
        """

        logger.info("Performing ANOVA.")

        # Configure groups and group-pair-indicator combinations.
        groups = set(self.lcps["slope"]["group"])
        group_indicator_combos = tuple(product(groups, set(self.dependencies)))
        group_pair_indicator_combos = tuple(product(combinations(groups, 2), set(self.dependencies)))

        # ANOVA - Equally weighted models.
        self.dst_df_anova_equal = pd.DataFrame({col: [None] * len(group_pair_indicator_combos) for col in
                                                ("group_a", "group_b", "agg_indicator", "fstat", "pvalue")})

        # Iterate aggregated indicators and group combinations.
        for iter_params in tqdm(group_pair_indicator_combos, desc="Performing ANOVA for equally weighted models"):
            group_a, group_b, agg_indicator = iter_params[0][0], iter_params[0][1], iter_params[1]

            # Perform ANOVA.
            anova = anova_lm(self.models_equal[group_a][agg_indicator],
                             self.models_equal[group_b][agg_indicator], test="F", typ=1)

            # Store results.
            flag_record = (self.dst_df_anova_equal["group_a"] == group_a) & \
                          (self.dst_df_anova_equal["group_b"] == group_b) & \
                          (self.dst_df_anova_equal["agg_indicator"] == agg_indicator)
            self.dst_df_anova_equal.loc[flag_record, "fstat"] = anova.iloc[1]["F"]
            self.dst_df_anova_equal.loc[flag_record, "pvalue"] = anova.iloc[1]["Pr(>F)"]

        # ANOVA - Regression-optimized models.
        self.dst_df_anova_optimized = pd.DataFrame({col: [None] * len(group_pair_indicator_combos) for col in
                                                    ("group_a", "group_b", "agg_indicator", "fstat", "pvalue")})

        # Iterate aggregated indicators and group combinations.
        for iter_params in tqdm(group_pair_indicator_combos, desc="Performing ANOVA for regression-optimized models"):
            group_a, group_b, agg_indicator = iter_params[0][0], iter_params[0][1], iter_params[1]

            # Perform ANOVA.
            anova = anova_lm(self.models_optimized[group_a][agg_indicator],
                             self.models_optimized[group_b][agg_indicator], test="F", typ=1)

            # Store results.
            flag_record = (self.dst_df_anova_optimized["group_a"] == group_a) & \
                          (self.dst_df_anova_optimized["group_b"] == group_b) & \
                          (self.dst_df_anova_optimized["agg_indicator"] == agg_indicator)
            self.dst_df_anova_optimized.loc[flag_record, "fstat"] = anova.iloc[1]["F"]
            self.dst_df_anova_optimized.loc[flag_record, "pvalue"] = anova.iloc[1]["Pr(>F)"]

        # ANOVA - Between models.
        self.dst_df_anova_between = pd.DataFrame({col: [None] * len(group_pair_indicator_combos) for col in
                                                  ("group", "agg_indicator", "fstat", "pvalue")})

        # Iterate aggregated indicators and groups.
        for iter_params in tqdm(group_indicator_combos, desc="Performing ANOVA between models"):
            group, agg_indicator = iter_params

            # Perform ANOVA.
            anova = anova_lm(self.models_equal[group][agg_indicator],
                             self.models_optimized[group][agg_indicator], test="F", typ=1)

            # Store results.
            flag_record = (self.dst_df_anova_between["group"] == group) & \
                          (self.dst_df_anova_between["agg_indicator"] == agg_indicator)
            self.dst_df_anova_between.loc[flag_record, "fstat"] = anova.iloc[1]["F"]
            self.dst_df_anova_between.loc[flag_record, "pvalue"] = anova.iloc[1]["Pr(>F)"]

    def gen_equations(self, equal_coeff: bool = False) -> pd.DataFrame:
        """
        Creates an equation for each group and aggregated manoeuvrability indicator.

        :param bool equal_coeff: Indicates if coefficients are to be equal. Default = False.
        :return pd.DataFrame: DataFrame containing equation components and evaluation metrics.
        """

        # Configure output DataFrame.
        groups, agg_indicators = zip(*product(set(self.lcps["slope"]["group"]), set(self.dependencies)))
        dst_cols = ("r2", "mae", "const", "slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds",
                    "machine_gun_viewsheds")
        results = pd.DataFrame({"group": groups, "agg_indicator": agg_indicators,
                                **{col: [None] * len(groups) for col in dst_cols}})

        # Iterate aggregated indicators.
        for agg_indicator, indicators in self.dependencies.items():

            # Iterate groups.
            for group in tqdm(sorted(set(self.lcps["slope"]["group"])),
                              desc=f"Generating equations for agg. indicator = {agg_indicator}; equal weight = "
                                   f"{equal_coeff}"):

                # Compile cost values for dependent and independent variables.
                flag_group = self.lcps["slope"]["group"] == group
                dependent = pd.Series(self.lcps[agg_indicator].loc[flag_group, "cost"])

                # Independent variables - equally weighted.
                if equal_coeff:

                    independent = pd.DataFrame({i: self.lcps[i].loc[flag_group, "cost"] for i in indicators})\
                        .sum(axis=1)

                # Independent variables - regression-optimized.
                else:

                    # Filter indicators to those that are statistically significant for group.
                    indicators_ = [i for i in indicators if group in self.pvalues[i]]
                    independent = pd.DataFrame({i: self.lcps[i].loc[flag_group, "cost"] for i in indicators_})

                # Add constant and fit regression model.
                independent = add_constant(independent)
                model = OLS(endog=dependent, exog=independent).fit()

                # Store model for ANOVA usage.
                if equal_coeff:
                    self.models_equal[group][agg_indicator] = model
                else:
                    self.models_optimized[group][agg_indicator] = model

                # Add equation coefficients and constant to results.
                flag_record = (results["group"] == group) & (results["agg_indicator"] == agg_indicator)

                # Coefficients - equally weighted.
                if equal_coeff:
                    results.loc[flag_record, "const"] = round(model.params["const"], 4)
                    for indicator in indicators:
                        results.loc[flag_record, indicator] = round(model.params[0], 4)

                # Coefficients - regression-optimized.
                else:
                    for name, coefficient in model.params.items():
                        results.loc[flag_record, name] = round(coefficient, 4)

                # Add r-squared to results.
                results.loc[flag_record, "r2"] = round(model.rsquared, 4)

                # Add mean absolute error to results.
                predicted = model.predict(independent)
                results.loc[flag_record, "mae"] = round(np.mean(np.abs(dependent - predicted)), 4)

        return results


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
def main(src: Path, pvalue_slope: Path, pvalue_ground_conditions: Path, pvalue_avenues_of_approach: Path,
         pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path) -> None:
    """
    \b
    Description: Creates a regression-optimized and equally weighted regression model for each group and aggregated
    manoeuvrability indicator using multiple linear regression from least-cost path cost values whereby:
        - independent variables: manoeuvrability indicators.
        - dependent variable: aggregated manoeuvrability indicator.
    The regression-optimized models will use only those indicators that are statistically significant for that group
    and aggregated indicator; equally weighted models will sum all indicators used for the given aggregated indicator
    to produce models with identical coefficients.

    Regression model outputs are used to perform Analysis of Variance (ANOVA) for each group and aggregated
    manoeuvrability indicator for each of the following:
        1. ANOVA between each group combination using equally weighted models.
        2. ANOVA between each group combination using regression-optimized models.
        3. ANOVA between the equally weighted and regression-optimized models for each group.

    \b
    Output: Outputs five .csv files within the same directory the source pvalue CSVs:
        1. equations_equally_weighted.csv: Equally weighted equations and evaluation metrics.
        2. equations_regression_optimized.csv: Regression-optimized equations and evaluation metrics.
        3. anova_equally_weighted.csv: ANOVA results for equally weighted models.
        3. anova_regression_optimized.csv: ANOVA results for regression-optimized models.
        3. anova_between.csv: ANOVA results for equally weighted vs. regression-optimized models.

    Each equations output file will contain the following attributes:
        - group: Group name.
        - agg_indicator: Name of the aggregated manoeuvrability indicator that the equation is for.
        - r2: Coefficient of determination for equation.
        - mae: Mean absolute error for equation.
        - const: Y-intercept of equation.
        - slope: Coefficient for slope variable.
        - ground_conditions: Coefficient for ground_conditions variable.
        - avenues_of_approach: Coefficient for avenues_of_approach variable.
        - rifle_viewsheds: Coefficient for rifle_viewsheds variable.
        - machine_gun_viewsheds: Coefficient for machine_gun_viewsheds variable.

    Each ANOVA output file will contain the following attributes:
        - group: Group name (anova_between only).
        - group_a: First group name (anova_equally_weighted and anova_regression_optimized only).
        - group_b: Second group name (anova_equally_weighted and anova_regression_optimized only).
        - agg_indicator: Name of the aggregated manoeuvrability indicator that the ANOVA results are for.
        - fstat: F-statistic.
        - pvalue: P-value for significance testing.

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
    :param Path pvalue_ground_conditions: CSV (.csv) containing pvalues for indicator=ground conditions.
    :param Path pvalue_avenues_of_approach: CSV (.csv) containing pvalues for indicator=avenues of approach.
    :param Path pvalue_rifle_viewsheds: CSV (.csv) containing pvalues for indicator=rifle viewsheds.
    :param Path pvalue_machine_gun_viewsheds: CSV (.csv) containing pvalues for indicator=machine gun viewsheds.
    """

    try:

        regression_anova = RegressionAnova(src, pvalue_slope, pvalue_ground_conditions, pvalue_avenues_of_approach,
                                           pvalue_rifle_viewsheds, pvalue_machine_gun_viewsheds)
        regression_anova()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
