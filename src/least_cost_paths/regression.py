import click
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
from collections import defaultdict
from itertools import chain, combinations, product
from pathlib import Path
from pingouin import pairwise_gameshowell
from scipy.stats import ttest_ind
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools import add_constant
from tqdm import tqdm

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class RidgeOLS:
    """Imitates a statsmodels OLS model for an sklearn RidgeCV model."""

    def __init__(self, ridge_model: RidgeCV, dependent: pd.Series, independent: pd.DataFrame) -> None:
        """
        Initializes the OLS imitation class.

        :param RidgeCV ridge_model: RidgeCV model.
        :param pd.Series dependent: Dependent variable values.
        :param pd.DataFrame independent: Independent variable values.
        """

        self.params = pd.Series(np.hstack([ridge_model.intercept_, ridge_model.coef_]),
                                index=["const", *independent.columns])
        self.nobs = len(dependent)
        self.df_model = independent.shape[1]
        self.df_resid = self.nobs - (self.df_model + 1)
        self.fitted_values = ridge_model.predict(independent)
        self.residuals = dependent - self.fitted_values
        self.ssr = np.sum(self.resid ** 2)
        self.sst = np.sum((dependent - np.mean(dependent)) ** 2)
        self.rsquared = 1 - (self.ssr / self.sst)
        self.scale = np.sum(self.resid ** 2) / self.df_resid

    @property
    def fittedvalues(self) -> pd.Series:
        """Imitates the OLS.fittedvalues property."""

        return self.fitted_values

    @property
    def model(self):
        """Imitates the OLS.model property."""

        return self

    @property
    def resid(self) -> pd.Series:
        """Imitates the OLS.resid property."""

        return self.residuals

    def predict(self, independent: pd.DataFrame) -> np.ndarray:
        """Imitates the OLS.predict method."""

        return np.dot(add_constant(independent), self.params)


class Regression:
    """Defines the Regression class."""

    def __init__(self, src: Path, pvalue_slope: Path, pvalue_ground_conditions: Path, pvalue_avenues_of_approach: Path,
                 pvalue_rifle_viewsheds: Path, pvalue_machine_gun_viewsheds: Path) -> None:
        """Initializes the Regression class."""

        self.dst_df_equations_equal = pd.DataFrame()
        self.dst_df_equations_optimized = pd.DataFrame()
        self.dst_equations_equal = Path(src.parent / "equations_equally_weighted.csv")
        self.dst_equations_optimized = Path(src.parent / "equations_regression_optimized.csv")
        self.dst_df_gh_equal = pd.DataFrame()
        self.dst_df_gh_optimized = pd.DataFrame()
        self.dst_df_welchs = pd.DataFrame()
        self.dst_gh_equal = Path(src.parent / "games_howell_equally_weighted.csv")
        self.dst_gh_optimized = Path(src.parent / "games_howell_regression_optimized.csv")
        self.dst_welchs = Path(src.parent / "welchs_ttest.csv")
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
        """Executes the Regression class."""

        # Create equations.
        self.dst_df_equations_equal = self.gen_equations(equal_coeff=True)
        self.dst_df_equations_optimized = self.gen_equations(equal_coeff=False)

        # Perform and compile results of Welch's t-test and Games-Howell test.
        self.welchs_ttest()
        self.games_howell()

        # Export results.
        self.dst_df_equations_equal.to_csv(self.dst_equations_equal, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_equations_equal}.")

        self.dst_df_equations_optimized.to_csv(self.dst_equations_optimized, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_equations_optimized}.")

        self.dst_df_gh_equal.to_csv(self.dst_gh_equal, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_gh_equal}.")

        self.dst_df_gh_optimized.to_csv(self.dst_gh_optimized, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_gh_optimized}.")

        self.dst_df_welchs.to_csv(self.dst_welchs, sep=",", header=True, index=False)
        logger.info(f"Exported results to: {self.dst_welchs}.")

    def games_howell(self) -> None:
        """
        Performs Games-Howell test between each group combination (pairwise) using equally weighted models and
        regression-optimized models, for each aggregated manoeuvrability indicator.
        """

        # Configure groups and group-pair-indicator combinations.
        groups = set(self.lcps["slope"]["group"])
        groups_indicator_combos = tuple(product(combinations(groups, 2), set(self.dependencies)))

        # Iterate model types.
        for model_type in ("equally weighted", "regression-optimized"):

            # Compile models.
            models = {"equally weighted": self.models_equal, "regression-optimized": self.models_optimized}[model_type]

            # Create output DataFrame.
            results = pd.DataFrame({
                "group_a": [vals[0][0] for vals in groups_indicator_combos],
                "group_b": [vals[0][1] for vals in groups_indicator_combos],
                "agg_indicator": [vals[1] for vals in groups_indicator_combos],
                **{col: [0.0] * len(groups_indicator_combos) for col in
                   ("mean_diff", "mean_se", "tstat", "pvalue", "ci_upper", "ci_lower", "cohens_d")}
            })

            # Iterate aggregated indicators and group combinations.
            for agg_indicator in tqdm(self.dependencies, desc=f"Performing Games-Howell test for {model_type} models"):

                # Generate single DataFrame containing model residuals (absolute values) for all groups.
                residuals = {group: np.abs(models[group][agg_indicator].resid) for group in groups}
                df_resid = pd.DataFrame({
                    "residuals": chain.from_iterable(residuals.values()),
                    "group": chain.from_iterable([group] * len(residuals[group]) for group in residuals)})

                # Perform Games-Howell test.
                gh_results = pairwise_gameshowell(data=df_resid, dv="residuals", between="group", effsize="cohen")

                # Store results.
                for group_pair in [vals[0] for vals in groups_indicator_combos if vals[1] == agg_indicator]:

                    flag_test = ((gh_results["A"] == group_pair[0]) & (gh_results["B"] == group_pair[1])) | \
                                ((gh_results["A"] == group_pair[1]) & (gh_results["B"] == group_pair[0]))

                    flag_dst = (results["group_a"] == group_pair[0]) & \
                               (results["group_b"] == group_pair[1]) & \
                               (results["agg_indicator"] == agg_indicator)

                    results.loc[flag_dst, "mean_diff"] = round(gh_results.loc[flag_test, "diff"].iloc[0], 4)
                    results.loc[flag_dst, "mean_se"] = round(gh_results.loc[flag_test, "se"].iloc[0], 4)
                    results.loc[flag_dst, "tstat"] = round(gh_results.loc[flag_test, "T"].iloc[0], 4)
                    results.loc[flag_dst, "pvalue"] = round(gh_results.loc[flag_test, "pval"].iloc[0], 4)
                    results.loc[flag_dst, "cohens_d"] = round(gh_results.loc[flag_test, "cohen"].iloc[0], 4)

                    # Calculate confidence intervals manually - store results.
                    df_, diff_, se_ = gh_results.loc[flag_test, ["df", "diff", "se"]].iloc[0].values
                    t_critical = stats.t.ppf(q=0.975, df=df_)
                    results.loc[flag_dst, "ci_lower"] = diff_ - (t_critical * se_)
                    results.loc[flag_dst, "ci_upper"] = diff_ + (t_critical * se_)

            # Store final results.
            if model_type == "equally weighted":
                self.dst_df_gh_equal = results.copy(deep=True)
            else:
                self.dst_df_gh_optimized = results.copy(deep=True)

    def gen_equations(self, equal_coeff: bool = False) -> pd.DataFrame:
        """
        Creates an equation for each group and aggregated manoeuvrability indicator. Uses OLS regression for equally
        weighted models and ridge regression for regression-optimized models.

        :param bool equal_coeff: Indicates if coefficients are to be equal. Default = False.
        :return pd.DataFrame: DataFrame containing equation components and evaluation metrics.
        """

        # Configure output DataFrame.
        groups, agg_indicators = zip(*product(set(self.lcps["slope"]["group"]), set(self.dependencies)))
        dst_cols = ("r2", "mae", "const", "slope", "ground_conditions", "avenues_of_approach", "rifle_viewsheds",
                    "machine_gun_viewsheds", "viewsheds")
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
                    independent = pd.DataFrame({"independent": independent})

                # Independent variables - regression-optimized.
                else:

                    # Filter indicators to those that are statistically significant for group.
                    indicators_ = [i for i in indicators if group in self.pvalues[i]]
                    independent = pd.DataFrame({i: self.lcps[i].loc[flag_group, "cost"] for i in indicators_})

                # Aggregate viewshed indicators via mean, if applicable to current aggregated indicator.
                # Note: Indicators adjustment recommended due to extreme multicollinearity after manual inspection.
                if agg_indicator == "manoeuvrability":
                    viewshed_indicators = [col for col in independent.columns if col.endswith("_viewsheds")]
                    if len(viewshed_indicators) > 1:

                        # Aggregate indicators.
                        independent["viewsheds"] = independent[viewshed_indicators].mean(axis=1)
                        independent.drop(columns=viewshed_indicators, inplace=True)

                # Regression - equally weighted.
                if equal_coeff:

                    # Add constant and fit regression model.
                    independent = add_constant(independent)
                    model = OLS(endog=dependent, exog=independent).fit()

                # Regression - regression-optimized.
                else:

                    # Create model training and testing data.
                    x_train, _, y_train, _ = train_test_split(independent, dependent, test_size=0.2,
                                                                        random_state=42, shuffle=True)
                    x_train = pd.DataFrame(x_train, columns=independent.columns)

                    # Determine best alpha value for ridge regression.
                    alpha_best = RidgeCV(alphas=np.logspace(start=-3, stop=3, num=100, endpoint=True),
                                         fit_intercept=True).fit(X=x_train, y=y_train).alpha_

                    # Fit ridge regression model.
                    model = RidgeCV(alphas=[alpha_best], fit_intercept=True)
                    model.fit(X=x_train, y=y_train)

                    # Convert RidgeCV model to OLS model.
                    model = RidgeOLS(ridge_model=model, dependent=dependent, independent=independent)

                # Store model for Games-Howell test usage.
                if equal_coeff:
                    self.models_equal[group][agg_indicator] = model
                else:
                    self.models_optimized[group][agg_indicator] = model

                # Add equation components to results.
                flag_record = (results["group"] == group) & (results["agg_indicator"] == agg_indicator)

                # Add constant, r-squared, and mean absolute error.
                results.loc[flag_record, "const"] = round(model.params["const"], 4)
                results.loc[flag_record, "r2"] = round(model.rsquared, 4)
                results.loc[flag_record, "mae"] = round(np.mean(np.abs(dependent - model.predict(independent))), 4)

                # Add coefficients - equally weighted.
                # Note: Conditionally populate aggregated viewsheds indicator, depending on the agg. indicator.
                if equal_coeff:
                    indicators_ = indicators
                    if agg_indicator == "manoeuvrability":
                        indicators_ = ["viewsheds" if i.endswith("_viewsheds") else i for i in indicators]
                    for name in indicators_:
                        results.loc[flag_record, name] = round(model.params["independent"], 4)

                # Add coefficients - regression-optimized.
                else:
                    for name, coefficient in model.params.items():
                        results.loc[flag_record, name] = round(coefficient, 4)

        return results

    def welchs_ttest(self) -> None:
        """
        Performs Welch's t-test between the equally weighted and regression-optimized models for each group and
        aggregated manoeuvrability indicator.
        """

        # Configure groups and group-indicator combinations.
        groups = set(self.lcps["slope"]["group"])
        group_indicator_combos = tuple(product(groups, set(self.dependencies)))

        # Create output DataFrame.
        self.dst_df_welchs = pd.DataFrame({
            "group": [vals[0] for vals in group_indicator_combos],
            "agg_indicator": [vals[1] for vals in group_indicator_combos],
            **{col: [0.0] * len(group_indicator_combos) for col in ("tstat", "pvalue")}
        })

        # Iterate aggregated indicators and groups.
        for iter_params in tqdm(group_indicator_combos, desc="Performing Welch's t-test"):
            group, agg_indicator = iter_params

            # Perform Welch's t-test.
            ttest = ttest_ind(np.abs(self.models_equal[group][agg_indicator].resid),
                              np.abs(self.models_optimized[group][agg_indicator].resid), equal_var=False)

            # Store results.
            flag_record = (self.dst_df_welchs["group"] == group) & \
                          (self.dst_df_welchs["agg_indicator"] == agg_indicator)
            self.dst_df_welchs.loc[flag_record, "tstat"] = ttest.statistic
            self.dst_df_welchs.loc[flag_record, "pvalue"] = ttest.pvalue

        # Apply Benjamini-Hochberg procedure for False Discovery Rate (BH-FDR) control.
        for agg_indicator in tqdm(set([vals[1] for vals in group_indicator_combos]), desc="Applying BH-FDR correction"):

            # Compile pvalues.
            flag_records = (self.dst_df_welchs["agg_indicator"] == agg_indicator) & \
                           (~self.dst_df_welchs["pvalue"].isna())
            pvalues = self.dst_df_welchs.loc[flag_records, "pvalue"].values

            # Apply BH-FDR control to get corrected pvalues.
            pvalues_corr = multipletests(pvalues, alpha=0.05, method="fdr_bh", is_sorted=False, returnsorted=False)[1]

            # Store corrected pvalues.
            self.dst_df_welchs.loc[flag_records, "pvalue"] = pvalues_corr


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

    \b
    Regression model outputs are used to perform the following:
        1. Welch's t-test between the equally weighted and regression-optimized models for each group and aggregated
           manoeuvrability indicator.
        2. Games-Howell Test between each group combination (pairwise) using equally weighted models for each
           aggregated manoeuvrability indicator.
        3. Games-Howell Test between each group combination (pairwise) using regression-optimized models for each
           aggregated manoeuvrability indicator.

    \b
    Output files: Outputs five .csv files within the same directory the source pvalue CSVs:
        1. equations_equally_weighted.csv: Equally weighted equations and evaluation metrics.
        2. equations_regression_optimized.csv: Regression-optimized equations and evaluation metrics.
        3. welchs_ttest.csv: Welch's t-test results.
        4. games_howell_equally_weighted.csv: Games-Howell test results for equally weighted models.
        5. games_howell_regression_optimized.csv: Games-Howell test results for regression-optimized models.

    \b
    Equation output file attributes:
        - group: Group name.
        - agg_indicator: Aggregated manoeuvrability indicator.
        - r2: Coefficient of determination for equation.
        - mae: Mean absolute error for equation.
        - const: Y-intercept of equation.
        - slope: Coefficient for slope variable.
        - ground_conditions: Coefficient for ground_conditions variable.
        - avenues_of_approach: Coefficient for avenues_of_approach variable.
        - rifle_viewsheds: Coefficient for rifle_viewsheds variable (except manoeuvrability agg. indicator).
        - machine_gun_viewsheds: Coefficient for machine_gun_viewsheds variable (except manoeuvrability agg. indicator).
        - viewsheds: Coefficient for mean-aggregated viewshed variables (manoeuvrability agg. indicator only).

    \b
    Welch's t-test output file attributes:
        - group: Group name.
        - agg_indicator: Aggregated manoeuvrability indicator.
        - tstat: t-test value.
        - pvalue: p-value for significance testing, corrected for False Discovery Rate via Benjamini-Hochberg procedure.

    \b
    Games-Howell test output file attributes:
        - group_a: First group name in the pairwise analysis.
        - group_b: Second group name in the pairwise analysis.
        - agg_indicator: Aggregated manoeuvrability indicator.
        - mean_diff: Difference in means between the groups.
        - mean_se: Standard error of the mean difference.
        - tstat: t-test value.
        - pvalue: Adjusted p-value for significance testing.
        - ci_upper: Upper value of 95% confidence interval.
        - ci_lower: Lower value of 95% confidence interval.
        - cohens_d: Effect size via Cohen's d.

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

        regression = Regression(src, pvalue_slope, pvalue_ground_conditions, pvalue_avenues_of_approach,
                                pvalue_rifle_viewsheds, pvalue_machine_gun_viewsheds)
        regression()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
