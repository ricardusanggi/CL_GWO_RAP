import math
import numpy as np
from mealpy.optimizer import Optimizer


class BFLGWO_YongWang(Optimizer):
    """
    BFLGWO:
    Binary Foraging-Levy Grey Wolf Optimizer

    Paper:
    Yong Wang et al. (2024)
    Role-oriented binary grey wolf optimizer using
    foraging-following and Lévy flight for feature selection
    """

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 50,
        **kwargs: object
    ) -> None:

        super().__init__(**kwargs)

        self.epoch = self.validator.check_int(
            "epoch",
            epoch,
            [1, 100000]
        )

        self.pop_size = self.validator.check_int(
            "pop_size",
            pop_size,
            [5, 10000]
        )

        self.set_parameters([
            "epoch",
            "pop_size"
        ])

        self.sort_flag = False

    ###########################################################################
    # LEVY FLIGHT
    ###########################################################################

    def levy_flight(self, dims, beta=1.5):

        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            /
            (
                math.gamma((1 + beta) / 2)
                * beta
                * (2 ** ((beta - 1) / 2))
            )
        ) ** (1 / beta)

        u = self.generator.normal(
            0,
            sigma,
            dims
        )

        v = self.generator.normal(
            0,
            1,
            dims
        )

        step = u / (
            np.abs(v) ** (1 / beta)
        )

        return step

    ###########################################################################
    # EVOLVE
    ###########################################################################

    def evolve(self, epoch):

        #######################################################################
        # a = 2 - 2(t/Tmax)
        #######################################################################

        a = 2 - 2.0 * epoch / self.epoch

        #######################################################################
        # TABLE 6 PARAMETERS
        #######################################################################

        v = a
        lambda_ = 2 / self.pop_size

        #######################################################################
        # GET ALPHA BETA DELTA
        #######################################################################

        _, list_best, _ = self.get_special_agents(
            self.pop,
            n_best=3,
            minmax=self.problem.minmax
        )

        alpha = list_best[0]
        beta = list_best[1]
        delta = list_best[2]

        #######################################################################
        # NEW POPULATION
        #######################################################################

        pop_new = []

        #######################################################################
        # LOOP
        #######################################################################

        for idx in range(0, self.pop_size):

            current = self.pop[idx]

            pos_current = current.solution.copy()

            ###################################################################
            # STEP 1
            # TOP WOLVES:
            # FORAGING + FOLLOWING
            ###################################################################

            if idx < 3:

                ###############################################################
                # FOLLOWING
                ###############################################################

                if idx == 0:
                    leader = beta
                else:
                    leader = alpha

                rand_follow = self.generator.random(
                    self.problem.n_dims
                )

                pos_current = (
                    pos_current
                    + rand_follow
                    * (
                        leader.solution
                        - pos_current
                    )
                )

                ###############################################################
                # FORAGING
                ###############################################################

                rand_forage = self.generator.random(
                    self.problem.n_dims
                )

                pos_current = (
                    pos_current
                    + v
                    * rand_forage
                    * (
                        alpha.solution
                        - pos_current
                    )
                )

            ###################################################################
            # STEP 2
            # ORDINARY WOLVES:
            # LEVY FLIGHT
            ###################################################################

            else:

                r = self.generator.random()

                ###############################################################
                # PAPER CONDITION
                ###############################################################

                if (a > 1) or (r < a):

                    levy = self.levy_flight(
                        self.problem.n_dims
                    )

                    pos_current = (
                        pos_current
                        + lambda_
                        * levy
                        * (
                            pos_current
                            - alpha.solution
                        )
                    )

            ###################################################################
            # STEP 3
            # STANDARD GWO UPDATE
            ###################################################################

            A1 = a * (
                2 * self.generator.random(
                    self.problem.n_dims
                ) - 1
            )

            A2 = a * (
                2 * self.generator.random(
                    self.problem.n_dims
                ) - 1
            )

            A3 = a * (
                2 * self.generator.random(
                    self.problem.n_dims
                ) - 1
            )

            C1 = 2 * self.generator.random(
                self.problem.n_dims
            )

            C2 = 2 * self.generator.random(
                self.problem.n_dims
            )

            C3 = 2 * self.generator.random(
                self.problem.n_dims
            )

            ###################################################################
            # X1
            ###################################################################

            X1 = (
                alpha.solution
                - A1
                * np.abs(
                    C1 * alpha.solution
                    - pos_current
                )
            )

            ###################################################################
            # X2
            ###################################################################

            X2 = (
                beta.solution
                - A2
                * np.abs(
                    C2 * beta.solution
                    - pos_current
                )
            )

            ###################################################################
            # X3
            ###################################################################

            X3 = (
                delta.solution
                - A3
                * np.abs(
                    C3 * delta.solution
                    - pos_current
                )
            )

            ###################################################################
            # FINAL POSITION
            ###################################################################

            pos_new = (
                X1 + X2 + X3
            ) / 3.0

            ###################################################################
            # BOUNDARY
            ###################################################################

            pos_new = self.correct_solution(
                pos_new
            )

            ###################################################################
            # CREATE AGENT
            ###################################################################

            agent = self.generate_empty_agent(
                pos_new
            )

            pop_new.append(agent)

            ###################################################################
            # SEQUENTIAL MODE
            ###################################################################

            if self.mode not in self.AVAILABLE_MODES:

                agent.target = self.get_target(
                    pos_new
                )

                self.pop[idx] = self.get_better_agent(
                    agent,
                    self.pop[idx],
                    self.problem.minmax
                )

        #######################################################################
        # PARALLEL MODE
        #######################################################################

        if self.mode in self.AVAILABLE_MODES:

            pop_new = self.update_target_for_population(
                pop_new
            )

            self.pop = self.greedy_selection_population(
                self.pop,
                pop_new,
                self.problem.minmax
            )