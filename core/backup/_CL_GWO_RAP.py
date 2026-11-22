class CL_GWO(Optimizer):
    """
    Comprehensive Learning Grey Wolf Optimizer (CL-GWO)

    Inspired by CL-PSO, this version introduces:
        - Dimension-wise learning
        - Random leader selection
        - Stagnation control via learning flags
    """

    def __init__(self, epoch=10000, pop_size=100,
                 w_min=0.2, w_max=2.0,
                 max_flag=7, **kwargs):
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.w_min = self.validator.check_float("w_min", w_min, (0, 1.0))
        self.w_max = self.validator.check_float("w_max", w_max, (1.0, 3.0))
        self.max_flag = self.validator.check_int("max_flag", max_flag, [2, 100])

        self.set_parameters(["epoch", "pop_size", "w_min", "w_max", "max_flag"])
        self.sort_flag = False
        self.is_parallelizable = True

    def initialize_variables(self):
        self.flags = np.zeros(self.pop_size)

    def evolve(self, epoch):
        # Adaptive control parameter
        a = self.w_max - (self.w_max - self.w_min) * (epoch / self.epoch)

        _, leaders, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        alpha, beta, delta = leaders

        pop_new = []

        for idx in range(self.pop_size):
            X_new = self.pop[idx].solution.copy()

            for d in range(self.problem.n_dims):

                # Learning probability
                pci = 0.05 + 0.45 * idx / self.pop_size

                if self.generator.random() > pci:
                    leader = alpha
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(self.pop_size)) - {idx}), 2, replace=False
                    )
                    leader = self.pop[id1] if self.compare_target(
                        self.pop[id1].target, self.pop[id2].target, self.problem.minmax
                    ) else self.pop[id2]

                A = a * (2 * self.generator.random() - 1)
                C = 2 * self.generator.random()

                D = abs(C * leader.solution[d] - self.pop[idx].solution[d])
                X_new[d] = leader.solution[d] - A * D

            X_new = self.correct_solution(X_new)
            agent = self.generate_empty_agent(X_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(X_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )

                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
                    if self.flags[idx] >= self.max_flag:
                        self.flags[idx] = 0

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )
