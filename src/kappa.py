import click
import logging
import numpy as np
import sys
from copy import deepcopy
from itertools import chain
from tabulate import tabulate
from tqdm import tqdm

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class Kappa:
    """Defines the Kappa class."""

    def __init__(self, classification: str, indicator_historical: str, indicator_analysis: str) -> None:
        """Initializes the Kappa class."""

        self.classification = classification
        self.indicator_historical = indicator_historical
        self.indicator_analysis = indicator_analysis

        # Define kappa, CI, and p-value variables.
        self.n_permutations = 9999
        self.p_exp = 0.0
        self.kappa = {
            "red": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "all": 0.0
        }
        self.ci = {
            "red": {"lower": 0.0, "upper": 0.0},
            "blue": {"lower": 0.0, "upper": 0.0},
            "green": {"lower": 0.0, "upper": 0.0},
            "all": {"lower": 0.0, "upper": 0.0}
        }
        self.pvalues = {
            "red": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "all": 0.0
        }

        # Define rating scales.
        self.scale = {
            "ordinal": list(range(3)),
            "binary": list(range(2))
        }[self.classification]

        # Define ratings.
        self.r_historical = {
            "ordinal": {
                "terrain_passability": {
                    "red": [2, 1, 0, 2, 2, 1],
                    "blue": [1, 1, 2, 2, 1, 0],
                    "green": [0, 0, 2, 0, 0, 0]
                },
                "hazard_exposure": {
                    "red": [2, 2, 1, 2, 1, 2],
                    "blue": [2, 2, 2, 1, 1, 0],
                    "green": [1, 1, 2, 2, 1, 1]
                },
                "manoeuvrability": {
                    "red": [2, 1, 1, 2, 1, 2],
                    "blue": [2, 2, 2, 2, 1, 0],
                    "green": [1, 1, 2, 1, 1, 1]
                }
            },
            "binary": {
                "terrain_passability": {
                    "red": [1, 0, 0, 1, 1, 0],
                    "blue": [1, 1, 1, 1, 0, 0],
                    "green": [0, 0, 1, 0, 0, 0]
                },
                "hazard_exposure": {
                    "red": [1, 1, 0, 1, 1, 1],
                    "blue": [1, 1, 1, 1, 0, 0],
                    "green": [0, 0, 1, 1, 0, 0]
                },
                "manoeuvrability": {
                    "red": [1, 1, 0, 1, 1, 1],
                    "blue": [1, 1, 1, 1, 0, 0],
                    "green": [0, 0, 1, 0, 0, 0]
                }
            }
        }[self.classification][self.indicator_historical]

        self.r_analysis = {
            "ordinal": {
                "terrain_passability": {
                    "red": [1, 0, 0, 1, 1, 1],
                    "blue": [1, 0, 2, 1, 1, 0],
                    "green": [0, 0, 1, 0, 0, 0]
                },
                "hazard_exposure": {
                    "red": [0, 2, 0, 0, 1, 0],
                    "blue": [2, 1, 2, 1, 1, 0],
                    "green": [1, 0, 1, 2, 1, 1]
                },
                "manoeuvrability": {
                    "red": [2, 0, 0, 1, 0, 0],
                    "blue": [1, 2, 2, 1, 0, 1],
                    "green": [1, 1, 0, 0, 1, 1]
                },
                "all": {
                    "red": [1, 1, 0, 1, 1, 1],
                    "blue": [1, 1, 2, 1, 1, 0],
                    "green": [1, 0, 0, 1, 0, 1]
                }
            },
            "binary": {
                "terrain_passability": {
                    "red": [1, 0, 0, 0, 0, 1],
                    "blue": [0, 0, 1, 1, 0, 0],
                    "green": [0, 0, 0, 0, 0, 0]
                },
                "hazard_exposure": {
                    "red": [0, 1, 0, 0, 0, 0],
                    "blue": [1, 1, 1, 1, 0, 0],
                    "green": [0, 0, 0, 1, 0, 1]
                },
                "manoeuvrability": {
                    "red": [1, 0, 0, 0, 0, 0],
                    "blue": [0, 1, 1, 1, 0, 0],
                    "green": [0, 0, 0, 0, 0, 0]
                },
                "all": {
                    "red": [1, 0, 0, 0, 0, 0],
                    "blue": [1, 1, 1, 1, 0, 0],
                    "green": [0, 0, 0, 0, 0, 0]
                }
            }
        }[self.classification][self.indicator_analysis]

    def __call__(self) -> None:
        """Executes the Kappa class."""

        self.kappa = self.calculate_kappa(r_historical=self.r_historical, r_analysis=self.r_analysis, reuse_pexp=False)
        self.calculate_ci()
        self.calculate_pvalue()

        # Log results.
        table = tabulate([[obj, kappa, self.pvalues[obj], self.ci[obj]["lower"], self.ci[obj]["upper"]] for obj, kappa
                          in self.kappa.items()],
                         headers=["Objective Line", "kappa", "pvalue", "ci_lower", "ci_upper"], tablefmt="rst",
                         colalign=("left", "right", "right", "right", "right"))
        logger.info("\n" + table)

    def calculate_ci(self) -> None:
        """Calculates 95% confidence intervals for Cohen's kappa coefficients using bootstrap resampling methodology."""

        logger.info("Calculating 95% confidence intervals via bootstrap method.")

        # Iteratively re-calculate kappa.
        kappa_bootstrap = {k: list() for k in self.kappa}
        for _ in tqdm(range(self.n_permutations), desc="Calculating kappa for resampled ratings"):

            # Compile rating pairs.
            pairs = list(zip(list(chain.from_iterable(self.r_historical.values())),
                             list(chain.from_iterable(self.r_analysis.values()))))

            # Resample rating pairs.
            resampled_idxs = np.random.choice(len(pairs), size=len(pairs), replace=True)
            resampled_pairs = [pairs[resampled_idx] for resampled_idx in resampled_idxs]

            # Iteratively recompile results by objective lines.
            r_historical = {k: list() for k in self.r_historical}
            r_analysis = {k: list() for k in self.r_analysis}
            for idx, obj in enumerate(self.r_historical):

                # Store resampled rating pairs.
                r_historical[obj] = [pair[0] for pair in resampled_pairs[(idx * 6) : ((idx + 1) * 6)]]
                r_analysis[obj] = [pair[1] for pair in resampled_pairs[(idx * 6) : ((idx + 1) * 6)]]

            # Calculate kappa.
            kappa_results = self.calculate_kappa(r_historical=r_historical, r_analysis=r_analysis, reuse_pexp=True,
                                                 suppress_log=True)
            for obj, kappa in kappa_results.items():
                kappa_bootstrap[obj].append(kappa)

        # Calculate 95% confidence interval.
        for obj, kappa_obs in self.kappa.items():

            kappa_bootstrap_ = np.sort(np.array(kappa_bootstrap[obj]))
            ci_lower = np.percentile(kappa_bootstrap_, 2.5)
            ci_upper = np.percentile(kappa_bootstrap_, 97.5)
            self.ci[obj] = {"lower": ci_lower, "upper": ci_upper}

    def calculate_kappa(self, r_historical: dict[str, list[int]], r_analysis: dict[str, list[int]],
                        reuse_pexp: bool=False, suppress_log: bool=False) -> dict[str, float]:
        """
        Calculates Cohen's kappa coefficient using spatial adjacency weighting with partial matching.

        :param dict[str, list[int]] r_historical: Ratings for historical accounts.
        :param dict[str, list[int]] r_analysis: Ratings for manoeuvrability analysis.
        :param bool reuse_pexp: Indicates if expected alignment (p_exp) should be reused, rather than re-calculated.
        :param bool suppress_log: Indicates if logging should be suppressed (useful when repeatedly called).
        :return dict[str, float]: Cohen's kappa coefficients.
        """

        if not suppress_log:
            logger.info("Calculating Cohen's kappa coefficient (observed).")

        # Calculate score.
        score = {k: list() for k in set(self.kappa) - {"all"}}
        for obj in score:
            for idx in range(6):

                # Partial matching based on value.
                value_score = 1 - abs((r_historical[obj][idx] - r_analysis[obj][idx]) / (len(self.scale) - 1))

                # Partial matching based on spatial adjacency.
                if r_historical[obj][idx] == r_analysis[obj][idx]:
                    spatial_score = 1

                else:

                    # First exterior corridor.
                    if idx == 0:
                        spatial_score = 0.5 if (r_historical[obj][idx] == r_analysis[obj][idx + 1]) else 0

                    # Last exterior corridor.
                    elif idx == 5:
                        spatial_score = 0.5 if (r_historical[obj][idx] == r_analysis[obj][idx - 1]) else 0

                    # Interior corridors.
                    else:
                        spatial_score = 0.5 if ((r_historical[obj][idx] == r_analysis[obj][idx - 1]) or
                                                (r_historical[obj][idx] == r_analysis[obj][idx + 1])) else 0

                score[obj].append(max([value_score, spatial_score]))

        # Calculate observed alignment.
        p_obs = {k: 0.0 for k in self.kappa}
        for obj in p_obs:
            if obj == "all":
                p_obs[obj] = sum([sum(scores) for scores in score.values()]) / 18
            else:
                p_obs[obj] = sum(score[obj]) / 6

        # Calculate value proportions.
        prop_historical = {k: 0.0 for k in self.scale}
        for val in prop_historical:
            prop_historical[val] = sum([r_historical[obj].count(val) for obj in r_historical]) / 18

        prop_analysis = {k: 0.0 for k in self.scale}
        for val in prop_analysis:
            prop_analysis[val] = sum([r_analysis[obj].count(val) for obj in r_analysis]) / 18

        # Calculate expected alignment.
        if not reuse_pexp:
            if self.classification == "ordinal":

                for idx, val in enumerate(self.scale):

                    if idx == 0:
                        self.p_exp += ((prop_historical[val] * prop_analysis[val] * 1) +
                                       (prop_historical[val] * prop_analysis[self.scale[idx + 1]] * 0.5))

                    elif val == self.scale[-1]:
                        self.p_exp += ((prop_historical[val] * prop_analysis[val] * 1) +
                                       (prop_historical[val] * prop_analysis[self.scale[idx - 1]] * 0.5))

                    else:
                        self.p_exp += ((prop_historical[val] * prop_analysis[val] * 1) +
                                       (prop_historical[val] * prop_analysis[self.scale[idx - 1]] * 0.5) +
                                       (prop_historical[val] * prop_analysis[self.scale[idx + 1]] * 0.5))

            else:
                self.p_exp = sum([prop_historical[val] * prop_analysis[val] for val in self.scale])

        # Calculate kappa.
        kappa = dict()
        for obj in self.kappa:
            kappa[obj] = (p_obs[obj] - self.p_exp) / (1 - self.p_exp)

        return deepcopy(kappa)

    def calculate_pvalue(self) -> None:
        """Calculates one-tailed p-values for permuted Cohen's kappa coefficients."""

        logger.info("Calculating p-values using permutations.")

        # Iterate permutations.
        kappa_permutations = {k: list() for k in self.kappa}
        for _ in tqdm(range(self.n_permutations), desc="Calculating kappa for permuted ratings"):

            # Permute historical ratings set.
            ratings = list(chain.from_iterable(self.r_historical.values()))
            r_permuted = list(np.random.permutation(ratings))

            # Iteratively recompile results by objective line.
            r_historical = {k: list() for k in self.r_historical}
            for idx, obj in enumerate(r_historical):
                r_historical[obj] = r_permuted[(idx * 6): ((idx + 1) * 6)]

            # Calculate kappa.
            kappa_results = self.calculate_kappa(r_historical=r_historical, r_analysis=self.r_analysis, reuse_pexp=True,
                                                 suppress_log=True)
            for obj, kappa in kappa_results.items():
                kappa_permutations[obj].append(kappa)

        # Calculate p-value (one-tailed).
        for obj, kappa_obs in self.kappa.items():
            self.pvalues[obj] = (np.sum(np.array(kappa_permutations[obj]) >= kappa_obs) + 1) / (self.n_permutations + 1)


@click.command()
@click.argument("classification", type=click.Choice(["ordinal", "binary"], case_sensitive=False))
@click.argument("indicator_historical", type=click.Choice(["terrain_passability", "hazard_exposure", "manoeuvrability"],
                                                          case_sensitive=False))
@click.argument("indicator_analysis", type=click.Choice(["terrain_passability", "hazard_exposure", "manoeuvrability",
                                                         "all"], case_sensitive=False))
def main(classification: str, indicator_historical: str, indicator_analysis: str) -> None:
    """
    \b
    Description: Calculates Cohen's kappa coefficient, p-value (one-tailed), and 95% confidence intervals for the set
    of ratings from historical accounts and manoeuvrability analysis per objective line (red, blue, green, and
    cumulative 'all').

    \b
    Output: Metrics printed to console.

    \b
    Assumptions:
        - Each classification scheme and aggregated indicator has the same number of ratings per objective line.
        - There are 3 objective lines, each having 6 ratings per rater.

    \b
    :param str classification: Classification scheme of rating values.
    :param str indicator_historical: Aggregated indicator of the historical accounts ratings.
    :param str indicator_analysis: Aggregated indicator of the manoeuvrability analysis ratings.
    """

    try:

        kappa = Kappa(classification, indicator_historical, indicator_analysis)
        kappa()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
