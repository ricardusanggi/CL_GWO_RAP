class WS_GWO(Optimizer):
    """
    Weighted-Strata Grey Wolf Optimizer (WS-GWO)

    Hierarki kawanan serigala (dari terbaik ke terburuk):
      - Alpha, Beta, Delta  : pemimpin kawanan (top 3, referensi posisi)
      - Omega I   (top 25%) : anggota kawanan terbaik, eksploitasi kuat
      - Omega II  (25-50%)  : anggota kawanan menengah, seimbang
      - Omega III (50-75%)  : anggota kawanan bawah, mulai eksplorasi
      - Lone Wolf (bawah 25%): bukan kawanan, eksplorasi bebas penuh
    """

    STRATUM_OMEGA_I   = 0
    STRATUM_OMEGA_II  = 1
    STRATUM_OMEGA_III = 2
    STRATUM_LONE_WOLF = 3

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def _get_stratum_weights(self, stratum, epoch):
        t = epoch / self.epoch  # progress: 0.0 → 1.0

        if stratum == self.STRATUM_OMEGA_I:
            # Omega I: mengikuti alpha sangat kuat, makin konvergen di akhir
            w_alpha = 0.6 + 0.3 * t
            w_beta  = 0.3 * (1 - t)
            w_delta = 1.0 - w_alpha - w_beta
        elif stratum == self.STRATUM_OMEGA_II:
            # Omega II: seimbang, sedikit condong ke alpha
            w_alpha = 0.4 + 0.2 * t
            w_beta  = 0.35 - 0.1 * t
            w_delta = 1.0 - w_alpha - w_beta
        elif stratum == self.STRATUM_OMEGA_III:
            # Omega III: delta lebih dominan di awal, lalu menyeimbang
            w_alpha = 0.2 + 0.3 * t
            w_beta  = 0.3
            w_delta = 1.0 - w_alpha - w_beta
        else:
            # Lone Wolf: tidak terikat hierarki kawanan, bebas menjelajah
            w_alpha = 0.1 + 0.1 * t
            w_beta  = 0.15
            w_delta = 1.0 - w_alpha - w_beta

        w_delta = max(w_delta, 0.0)
        total = w_alpha + w_beta + w_delta
        return w_alpha / total, w_beta / total, w_delta / total

    def evolve(self, epoch):
        a = 2 - 2 * epoch / self.epoch
        _, leaders, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        alpha, beta, delta = leaders

        # Urutkan populasi berdasarkan fitness untuk penentuan stratum
        if self.problem.minmax == "min":
            sorted_indices = sorted(range(self.pop_size),
                                    key=lambda i: self.pop[i].target.fitness)
        else:
            sorted_indices = sorted(range(self.pop_size),
                                    key=lambda i: self.pop[i].target.fitness,
                                    reverse=True)

        # Batas 4 stratum (masing-masing 25% populasi)
        q1 = self.pop_size // 4
        q2 = self.pop_size // 2
        q3 = 3 * self.pop_size // 4

        stratum_map = {}
        for rank, idx in enumerate(sorted_indices):
            if rank < q1:
                stratum_map[idx] = self.STRATUM_OMEGA_I
            elif rank < q2:
                stratum_map[idx] = self.STRATUM_OMEGA_II
            elif rank < q3:
                stratum_map[idx] = self.STRATUM_OMEGA_III
            else:
                stratum_map[idx] = self.STRATUM_LONE_WOLF

        noise_table = {
            self.STRATUM_OMEGA_I:   0.00,
            self.STRATUM_OMEGA_II:  0.05,
            self.STRATUM_OMEGA_III: 0.10,
            self.STRATUM_LONE_WOLF: 0.30,
        }

        pop_new = []
        for i in range(self.pop_size):
            stratum = stratum_map[i]
            w_alpha, w_beta, w_delta = self._get_stratum_weights(stratum, epoch)
            a_eff = a * (1 + noise_table[stratum])

            A1 = a_eff * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a_eff * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a_eff * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)

            X1 = alpha.solution - A1 * abs(C1 * alpha.solution - self.pop[i].solution)
            X2 = beta.solution  - A2 * abs(C2 * beta.solution  - self.pop[i].solution)
            X3 = delta.solution - A3 * abs(C3 * delta.solution - self.pop[i].solution)

            pos_new = w_alpha * X1 + w_beta * X2 + w_delta * X3
            pos_new = self.correct_solution(pos_new)

            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[i] = self.get_better_agent(agent, self.pop[i], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
