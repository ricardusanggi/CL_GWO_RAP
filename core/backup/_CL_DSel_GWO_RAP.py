class CL_DSel_GWO(Optimizer):
    """
    Hybrid CL-GWO + Dimension-Selective GWO (hierarchy preserved)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        a = 2 - 2 * epoch / self.epoch
        _, leaders, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        alpha, beta, delta = leaders

        pop_new = []
        for i in range(self.pop_size):
            X_new = self.pop[i].solution.copy()

            for d in range(self.problem.n_dims):
                r = self.generator.random()
                if r < 0.5:
                    leader = alpha
                elif r < 0.8:
                    leader = beta
                else:
                    leader = delta

                A = a * (2 * self.generator.random() - 1)
                C = 2 * self.generator.random()

                X_new[d] = leader.solution[d] - A * abs(
                    C * leader.solution[d] - self.pop[i].solution[d]
                )

            X_new = self.correct_solution(X_new)
            agent = self.generate_empty_agent(X_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(X_new)
                self.pop[i] = self.get_better_agent(agent, self.pop[i], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
