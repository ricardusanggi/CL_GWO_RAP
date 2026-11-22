import numpy as np
from math import gamma as math_gamma
from mealpy.optimizer import Optimizer


class RLCGWO_ZhengpengHu(Optimizer):
    """
    Reinforcement Learning-based Comprehensive learning Grey Wolf Optimizer
    dari paper berjudul: Reinforcement learning-based comprehensive learning grey wolf optimizer
    (RLCGWO)
    https://doi.org/10.1016/j.asoc.2023.110959
    Zhengpeng Hu, Xiaobing Yu 
    
    Flow:
    - Chaotic Elite Learning Eq.(25) : leader_pos + rand * (2*zk - 1)
    - GWO base position               : X_gwo = X1 + (X2 + X3) / 3
    - RL SoftMax action selection
    - Exploitation Eq.(21)            : pbest + e * rand * (X_gwo - pbest)
    - Exploration  Eq.(15)            : Levy flight berbasis pbest
    - Q-table update (Bellman equation)

    Catatan: algoritma ini bersifat sequential — setiap agen memerlukan evaluasi
    fitness langsung untuk update Q-table, sehingga parallel mode tidak didukung.
    """

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        alpha_q=0.1,
        gamma=0.9,
        temperature=1.0,
        beta_levy=1.5,
        **kwargs
    ):

        super().__init__(**kwargs)

        # ==========================================
        # VALIDATED PARAMETERS (mealpy convention)
        # ==========================================
        self.epoch       = self.validator.check_int(  "epoch",       epoch,       [1, 100000])
        self.pop_size    = self.validator.check_int(  "pop_size",    pop_size,    [5, 10000])
        self.alpha_q     = self.validator.check_float("alpha_q",     alpha_q,     (0, 1.0))
        self.gamma       = self.validator.check_float("gamma",       gamma,       (0, 1.0))
        self.temperature = self.validator.check_float("temperature", temperature, (0, 10.0))
        self.beta_levy   = self.validator.check_float("beta_levy",   beta_levy,   (0, 2.0))

        self.set_parameters(
            ["epoch", "pop_size", "alpha_q", "gamma", "temperature", "beta_levy"]
        )

        # ==========================================
        # STATES : 0 -> success | 1 -> fail
        # ACTIONS: 0 -> exploitation | 1 -> exploration
        # ==========================================
        self.n_states  = 2
        self.n_actions = 2

        # Q-table: satu tabel dipakai bersama seluruh populasi
        self.Q_table = np.zeros((self.n_states, self.n_actions))

        # State RL saat ini untuk setiap wolf
        self.population_states = np.zeros(self.pop_size, dtype=int)

        # Personal best — diinisialisasi di initialization()
        self.pbest        = None
        self.pbest_target = None

        self.sort_flag = False

    # =========================================================
    # INITIALIZATION
    # Override parent: membuat populasi sekaligus inisialisasi pbest.
    # generate_population() sudah mengevaluasi target setiap agen.
    # =========================================================
    def initialization(self):

        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        self.pbest = [
            agent.solution.copy()
            for agent in self.pop
        ]

        self.pbest_target = [
            agent.target.fitness
            for agent in self.pop
        ]

    # =========================================================
    # LEVY FLIGHT (Mantegna algorithm)
    # Menggunakan self.generator agar reproducible dengan seed mealpy
    # =========================================================
    def levy_flight(self):

        sigma = (
            (
                math_gamma(1 + self.beta_levy)
                * np.sin(np.pi * self.beta_levy / 2)
            )
            /
            (
                math_gamma((1 + self.beta_levy) / 2)
                * self.beta_levy
                * 2 ** ((self.beta_levy - 1) / 2)
            )
        ) ** (1 / self.beta_levy)

        u = self.generator.normal(0, sigma, self.problem.n_dims)
        v = self.generator.normal(0, 1,     self.problem.n_dims)

        return u / (np.abs(v) ** (1 / self.beta_levy))

    # =========================================================
    # SOFTMAX ACTION SELECTION
    # =========================================================
    def softmax(self, q_values):

        exp_q = np.exp(q_values / self.temperature)

        return exp_q / np.sum(exp_q)

    # =========================================================
    # SELECT ACTION USING SOFTMAX
    # =========================================================
    def select_action(self, state):

        probs = self.softmax(self.Q_table[state])

        return self.generator.choice(
            np.arange(self.n_actions),
            p=probs
        )

    # =========================================================
    # UPDATE Q-TABLE (Bellman equation)
    # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s') - Q(s,a)]
    # =========================================================
    def update_q_table(self, state, action, reward, next_state):

        best_next = np.max(self.Q_table[next_state])

        self.Q_table[state, action] += self.alpha_q * (
            reward
            + self.gamma * best_next
            - self.Q_table[state, action]
        )

    # =========================================================
    # CHAOTIC ELITE LEARNING Eq.(25)
    # pos_new = leader_pos + rand * (2 * zk - 1)
    # zk: chaotic variable (random per call, bukan logistic map)
    # =========================================================
    def chaotic_elite_learning(self, leader_pos):

        zk   = self.generator.random(self.problem.n_dims)
        rand = self.generator.random(self.problem.n_dims)

        pos_new = leader_pos + rand * (2 * zk - 1)

        return self.correct_solution(pos_new)

    # =========================================================
    # EVOLVE — dipanggil setiap iterasi oleh mealpy
    # =========================================================
    def evolve(self, epoch):

        # =====================================================
        # PARAMETER a — menurun linier dari 2 ke 0
        # =====================================================
        a = 2 - 2 * epoch / self.epoch

        # =====================================================
        # GET LEADERS (alpha = terbaik, beta = ke-2, delta = ke-3)
        # =====================================================
        _, leaders, _ = self.get_special_agents(
            self.pop,
            n_best=3,
            minmax=self.problem.minmax
        )

        alpha = leaders[0]
        beta  = leaders[1]
        delta = leaders[2]

        pop_new = []

        # =====================================================
        # MAIN LOOP — setiap wolf diproses secara sequential
        # =====================================================
        for idx in range(self.pop_size):

            current = self.pop[idx]
            state   = self.population_states[idx]

            is_leader = (
                np.array_equal(current.solution, alpha.solution)
                or np.array_equal(current.solution, beta.solution)
                or np.array_equal(current.solution, delta.solution)
            )

            # =================================================
            # LEADER → CHAOTIC ELITE LEARNING Eq.(25)
            # =================================================
            if is_leader:

                pos_new = self.chaotic_elite_learning(current.solution)

            # =================================================
            # NON-LEADER → GWO + RL ACTION
            # =================================================
            else:

                # =============================================
                # Koefisien A dan C Eq.(23)
                # =============================================
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)

                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)

                D_alpha = np.abs(C1 * alpha.solution - current.solution)
                D_beta  = np.abs(C2 * beta.solution  - current.solution)
                D_delta = np.abs(C3 * delta.solution  - current.solution)

                # =============================================
                # Posisi kandidat dari tiap pemimpin Eq.(12)
                # =============================================
                X1 = alpha.solution - A1 * D_alpha
                X2 = beta.solution  - A2 * D_beta
                X3 = delta.solution - A3 * D_delta

                # =============================================
                # Agregasi posisi GWO
                # =============================================
                X_gwo = X1 + (X2 + X3) / 3.0

                # =============================================
                # RL ACTION SELECTION via SoftMax
                # =============================================
                action = self.select_action(state)

                # e: skala langkah, menurun seiring iterasi
                e = a / 2.0

                # =============================================
                # CASE 1 — EXPLOITATION Eq.(21)
                # pos = pbest + e * rand * (X_gwo - pbest)
                # =============================================
                if action == 0:

                    rand = self.generator.random(self.problem.n_dims)

                    pos_new = (
                        self.pbest[idx]
                        + e * rand * (X_gwo - self.pbest[idx])
                    )

                # =============================================
                # CASE 2 — EXPLORATION Eq.(15) via Levy flight
                # pos = pbest * LF * (current - pbest) * e
                # =============================================
                else:

                    LF = self.levy_flight()

                    pos_new = (
                        self.pbest[idx]
                        * LF
                        * (current.solution - self.pbest[idx])
                        * e
                    )

            # =================================================
            # BOUNDARY CONTROL
            # =================================================
            pos_new = self.correct_solution(pos_new)

            # =================================================
            # EVALUASI FITNESS (sequential — wajib langsung)
            # =================================================
            new_agent = self.generate_empty_agent(pos_new)
            new_agent.target = self.get_target(pos_new)

            # =================================================
            # GREEDY SELECTION
            # =================================================
            better = self.get_better_agent(
                new_agent,
                current,
                self.problem.minmax
            )

            # =================================================
            # REWARD & UPDATE PERSONAL BEST
            # reward=1 jika solusi baru lebih baik, reward=0 jika tidak
            # =================================================
            if better is new_agent:

                reward     = 1
                next_state = 0

                self.pbest[idx]        = new_agent.solution.copy()
                self.pbest_target[idx] = new_agent.target.fitness

            else:

                reward     = 0
                next_state = 1

            # =================================================
            # UPDATE Q-TABLE (hanya non-leader — leader tidak
            # memiliki action RL sehingga tidak ada Q-value yang perlu diperbarui)
            # =================================================
            if not is_leader:
                self.update_q_table(state, action, reward, next_state)

            # =================================================
            # UPDATE STATE RL UNTUK ITERASI BERIKUTNYA
            # =================================================
            self.population_states[idx] = next_state

            pop_new.append(better)

        # =====================================================
        # UPDATE POPULASI
        # =====================================================
        self.pop = pop_new
