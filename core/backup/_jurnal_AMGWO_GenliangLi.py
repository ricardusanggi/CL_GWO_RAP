import numpy as np
from mealpy.optimizer import Optimizer


class AMGWO_GenliangLi(Optimizer):
    """
    Adaptive Mechanism-based Grey Wolf Optimizer (AMGWO)

    Paper:
    Adaptive mechanism-based grey wolf optimizer
    for feature selection in high-dimensional classification
    https://doi.org/10.1371/journal.pone.0318903
    Genliang Li  ,Yaxin Cui ,Jingyu Su
    
    Main strategies:
    1. Adaptive nonlinear convergence factor
    2. Adaptive Fitness Distance Balance (AFDB)
    3. Adaptive neighborhood mutation
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int(
            "epoch", epoch, [1, 100000]
        )

        self.pop_size = self.validator.check_int(
            "pop_size", pop_size, [5, 10000]
        )

        self.set_parameters(["epoch", "pop_size"])

        self.sort_flag = False

    ###########################################################################
    # FITNESS DISTANCE BALANCE (FDB)
    ###########################################################################

    def get_fdb_agent(self):

        # Current fitness values
        fits = np.array([
            agent.target.fitness for agent in self.pop
        ])

        # Normalize fitness (minimization)
        fit_norm = (fits.max() - fits) / (
            fits.max() - fits.min() + 1e-10
        )

        # Distance to global best
        g_best = self.g_best.solution

        distances = np.array([
            np.linalg.norm(agent.solution - g_best)
            for agent in self.pop
        ])

        # Normalize distance
        dist_norm = distances / (
            distances.max() + 1e-10
        )

        # FDB score
        scores = fit_norm + dist_norm

        # Select highest FDB score
        idx = np.argmax(scores)

        return self.pop[idx]

    ###########################################################################
    # MAIN EVOLUTION
    ###########################################################################

    def evolve(self, epoch):

        #######################################################################
        # Eq. (10)
        # Nonlinear adaptive convergence factor
        #######################################################################

        a = 2 - 2 * np.sin(
            (np.pi / 2) * ((epoch / self.epoch) ** 3)
        )

        #######################################################################
        # Select alpha beta delta
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
        # FDB wolf
        #######################################################################

        fdb_agent = self.get_fdb_agent()

        pop_new = []

        #######################################################################
        # Main loop
        #######################################################################

        for idx in range(self.pop_size):

            current = self.pop[idx]

            ###################################################################
            # Eq. (14)
            # Adaptive FDB selection
            ###################################################################

            if epoch % 2 == 0:
                X_ref = current.solution
            else:
                X_ref = fdb_agent.solution

            ###################################################################
            # Coefficient vectors
            ###################################################################

            A1 = a * (
                2 * self.generator.random(self.problem.n_dims) - 1
            )

            A2 = a * (
                2 * self.generator.random(self.problem.n_dims) - 1
            )

            A3 = a * (
                2 * self.generator.random(self.problem.n_dims) - 1
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
            # Eq. (13)
            # Distance calculation
            ###################################################################

            D_alpha = np.abs(
                C1 * alpha.solution - X_ref
            )

            D_beta = np.abs(
                C2 * beta.solution - X_ref
            )

            D_delta = np.abs(
                C3 * delta.solution - X_ref
            )

            ###################################################################
            # Step update
            ###################################################################

            X1 = alpha.solution - A1 * D_alpha
            X2 = beta.solution - A2 * D_beta
            X3 = delta.solution - A3 * D_delta

            ###################################################################
            # Eq. (15)
            # Mutation intensity
            ###################################################################

            r1 = self.g_best.solution - X1
            r2 = self.g_best.solution - X2
            r3 = self.g_best.solution - X3

            ###################################################################
            # Eq. (16)
            # Adaptive neighborhood mutation
            ###################################################################

            X1 = X1 + r1 * self.generator.normal(
                0, 1, self.problem.n_dims
            )

            X2 = X2 + r2 * self.generator.normal(
                0, 1, self.problem.n_dims
            )

            X3 = X3 + r3 * self.generator.normal(
                0, 1, self.problem.n_dims
            )

            ###################################################################
            # Eq. (9)
            # Final position update
            ###################################################################

            pos_new = (X1 + X2 + X3) / 3.0

            ###################################################################
            # Boundary control
            ###################################################################

            pos_new = self.correct_solution(pos_new)

            ###################################################################
            # Create new agent
            ###################################################################

            agent = self.generate_empty_agent(pos_new)

            pop_new.append(agent)

            ###################################################################
            # Sequential evaluation
            ###################################################################

            if self.mode not in self.AVAILABLE_MODES:

                agent.target = self.get_target(pos_new)

                self.pop[idx] = self.get_better_agent(
                    agent,
                    self.pop[idx],
                    self.problem.minmax
                )

        #######################################################################
        # Parallel mode
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