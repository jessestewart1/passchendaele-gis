import click
import logging
import numpy as np
import pandas as pd
import sys
from itertools import product
from pathlib import Path
from sqlalchemy import create_engine
from statsmodels.api import OLS
from statsmodels.tools import add_constant

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
                 pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path) -> None:
        """Initializes the MultipleRegression class."""

        self.dst_equally = Path(src.parent / "equations_equally_weighted.csv")
        self.dst_regression = Path(src.parent / "equations_regression_weighted.csv")
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

            logger.info(f"Compiling least-cost paths: {src}, layer={layer}.")

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

        # Configure output DataFrame.
        groups, agg_indicators = zip(*product(set(self.lcps["slope"]["group"]), set(self.dependencies)))
        dst_cols = ("r2", "mae", "const", "slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds",
                    "machine_gun_viewsheds")
        self.dst_df_equally = pd.DataFrame({"group": groups, "agg_indicator": agg_indicators,
                                            **{col: [None] * len(groups) for col in dst_cols}})
        self.dst_df_regression = pd.DataFrame({"group": groups, "agg_indicator": agg_indicators,
                                               **{col: [None] * len(groups) for col in dst_cols}})

    def __call__(self) -> None:
        """Executes the MultipleRegression class."""

        self.gen_regression_equations()
        self.gen_equally_weighted_equations()

        # Export results.
        self.dst_df_equally.to_csv(self.dst_equally, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_equally}.")

        self.dst_df_regression.to_csv(self.dst_regression, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_regression}.")

    def gen_equally_weighted_equations(self) -> None:
        """Creates an equally weighted equation for each group and aggregated manoeuvrability indicator."""

        # Iterate groups.
        for group in sorted(set(self.lcps["slope"]["group"])):

            # Iterate aggregated indicators.
            for agg_indicator, indicators in self.dependencies.items():

                logger.info(f"Generating regression equation for group = {group}; "
                            f"aggregated indicator = {agg_indicator}.")

                # Compile cost values for dependent and independent variables.
                flag_group = self.lcps["slope"]["group"] == group
                independent = pd.DataFrame({i: self.lcps[i].loc[flag_group, "cost"] for i in indicators})
                dependent = pd.Series(self.lcps[agg_indicator].loc[flag_group, "cost"])

                # Populate results with equation components (equal weights).
                flag_record = ((self.dst_df_equally["group"] == group) &
                               (self.dst_df_equally["agg_indicator"] == agg_indicator))
                for indicator in indicators:
                    self.dst_df_equally.loc[flag_record, indicator] = round(1 / len(indicators), 4)

                # Add 0 constant to results.
                self.dst_df_equally.loc[flag_record, "const"] = 0

                # Calculate predicted values.
                multiplier = 1 / len(indicators)
                predicted = independent.apply(lambda row: sum([v * multiplier for v in row.values]), axis=1)

                # Add r-squared to results.
                r2 = 1 - (sum((dependent - predicted) ** 2) / sum((dependent - dependent.mean()) ** 2))
                self.dst_df_equally.loc[flag_record, "r2"] = round(r2, 4)

                # Add mean absolute error to results.
                self.dst_df_equally.loc[flag_record, "mae"] = np.mean(np.abs(dependent - predicted))

    def gen_regression_equations(self) -> None:
        """Creates a multiple regression model equation for each group and aggregated manoeuvrability indicator."""

        # Iterate groups.
        for group in sorted(set(self.lcps["slope"]["group"])):

            # Iterate aggregated indicators.
            for agg_indicator, indicators in self.dependencies.items():

                logger.info(f"Generating regression equation for group = {group}; "
                            f"aggregated indicator = {agg_indicator}.")

                # Filter indicators to those that are statistically significant for group.
                indicators = [i for i in indicators if group in self.pvalues[i]]

                # Compile cost values for dependent and independent variables.
                flag_group = self.lcps["slope"]["group"] == group
                independent = pd.DataFrame({i: self.lcps[i].loc[flag_group, "cost"] for i in indicators})
                dependent = pd.Series(self.lcps[agg_indicator].loc[flag_group, "cost"])

                # Add constant and fit regression model.
                independent = add_constant(independent)
                model = OLS(endog=dependent, exog=independent).fit()

                # Populate results with equation components.
                flag_record = ((self.dst_df_regression["group"] == group) &
                               (self.dst_df_regression["agg_indicator"] == agg_indicator))
                for name, coefficient in model.params.items():
                    self.dst_df_regression.loc[flag_record, name] = round(coefficient, 4)

                # Add r-squared to results.
                self.dst_df_regression.loc[flag_record, "r2"] = round(model.rsquared, 4)

                # Add mean absolute error to results.
                predicted = model.predict(independent)
                self.dst_df_regression.loc[flag_record, "mae"] = np.mean(np.abs(dependent - predicted))


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
    Description: For each group and aggregated manoeuvrability indicator, using the statistically significant
    manoeuvrability indicators specific to each group, creates a multiple linear regression equation from least-cost
    paths cost values whereby:
        - independent variable(s): manoeuvrability indicator(s).
        - dependent variable: aggregated manoeuvrability indicator.

    \b
    Output: Outputs two .csv files within the same directory the source pvalue CSVs:
        - equations_equally_weighted.csv: Equally weighted equations (non regression).
        - equations_regression_weighted.csv: Multiple linear regression-weighted equations.
    Each output file will contain the following attributes:
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
    """

    try:

        multiple_regression = MultipleRegression(src, pvalue_slope, pvalue_ground_conditions,
                                                 pvalue_avenues_of_approach, pvalue_rifle_viewsheds,
                                                 pvalue_machine_gun_viewsheds)
        multiple_regression()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
