import click
import logging
import numpy as np
import pandas as pd
import sys

# Set logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


class KappaBootstrap:
    """Defines the KappaBootstrap class."""

    def __init__(self, classification: str, indicator_historical: str, indicator_analysis: str) -> None:
        """Initializes the KappaBootstrap class."""

        self.classification = classification
        self.indicator_historical = indicator_historical
        self.indicator_analysis = indicator_analysis

        # Define kappa variables.
        self.p_obs = {
            "red": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "all": 0.0
        }
        self.p_exp = 0.0
        self.kappa = {
            "red": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "all": 0.0
        }
        self.n_permutations = 10000
        self.p_value = 0.0

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
        """Executes the KappaBootstrap class."""

        self.calculate_kappa()
        print(self.kappa)
        self.calculate_pvalue()

    def calculate_kappa(self) -> None:
        """Calculates Cohen's kappa coefficient using spatial adjacency weighting with partial matching."""

        logger.info("Calculating Cohen's kappa coefficient (observed).")

        # Calculate score.
        score = {k: list() for k in ("red", "blue", "green")}
        for obj in score:
            for idx in range(6):

                # Partial matching based on value.
                value_score = 1 - abs((self.r_historical[obj][idx] - self.r_analysis[obj][idx]) / (len(self.scale) - 1))

                # Partial matching based on spatial adjacency.
                if self.r_historical[obj][idx] == self.r_analysis[obj][idx]:
                    spatial_score = 1

                else:

                    # First exterior corridor.
                    if idx == 0:
                        spatial_score = 0.5 if (self.r_historical[obj][idx] == self.r_analysis[obj][idx + 1]) else 0

                    # Last exterior corridor.
                    elif idx == 5:
                        spatial_score = 0.5 if (self.r_historical[obj][idx] == self.r_analysis[obj][idx - 1]) else 0

                    # Interior corridors.
                    else:
                        spatial_score = 0.5 if ((self.r_historical[obj][idx] == self.r_analysis[obj][idx - 1]) or
                                                (self.r_historical[obj][idx] == self.r_analysis[obj][idx + 1])) else 0

                score[obj].append(max([value_score, spatial_score]))

        # Calculate observed alignment.
        for obj in self.p_obs:
            if obj == "all":
                self.p_obs[obj] = sum([sum(scores) for scores in score.values()]) / 18
            else:
                self.p_obs[obj] = sum(score[obj]) / 6

        # Calculate value proportions.
        prop_historical = {k: 0.0 for k in self.scale}
        for val in prop_historical:
            prop_historical[val] = sum([self.r_historical[obj].count(val) for obj in self.r_historical]) / 18

        prop_analysis = {k: 0.0 for k in self.scale}
        for val in prop_analysis:
            prop_analysis[val] = sum([self.r_analysis[obj].count(val) for obj in self.r_analysis]) / 18

        # Calculate expected alignment.
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
        for obj in self.kappa:
            self.kappa[obj] = (self.p_obs[obj] - self.p_exp) / (1 - self.p_exp)

    def calculate_pvalue(self) -> None:
        """Calculates empirical p-value for Cohen's kappa coefficient using bootstrap resampling methodology."""

        # TODO


@click.command()
@click.argument("classification", type=click.Choice(["ordinal", "binary"], case_sensitive=False))
@click.argument("indicator_historical", type=click.Choice(["terrain_passability", "hazard_exposure", "manoeuvrability"],
                                                          case_sensitive=False))
@click.argument("indicator_analysis", type=click.Choice(["terrain_passability", "hazard_exposure", "manoeuvrability",
                                                         "all"], case_sensitive=False))
def main(classification: str, indicator_historical: str, indicator_analysis: str) -> None:
    """
    \b
    Description: TODO.

    \b
    Assumptions:
        - TODO

    \b
    :param str classification: Classification scheme of rating values.
    :param str indicator_historical: Aggregated indicator of the historical accounts ratings.
    :param str indicator_analysis: Aggregated indicator of the manoeuvrability analysis ratings.
    """

    try:

        kappa_bootstrap = KappaBootstrap(classification, indicator_historical, indicator_analysis)
        kappa_bootstrap()

    except KeyboardInterrupt:
        logger.exception("KeyboardInterrupt: Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()
