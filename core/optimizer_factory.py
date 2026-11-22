#path: full_code/core/optimizer_factory.py
import numpy as np
import random

from mealpy.swarm_based.PSO import OriginalPSO, AIW_PSO, LDW_PSO, CLPSO_Liang, CLPSO_RAP


class PatchedCLPSO_Liang(CLPSO_Liang):
    """
    Workaround for mealpy 3.0.3 bug: get_better_agent may return an agent created by
    generate_empty_agent which lacks local_target, causing AttributeError in evolve().
    Fix: after greedy selection, ensure local_target is always present on self.pop[idx].
    """

    def _ensure_local_target(self, idx, fallback_target):
        if not hasattr(self.pop[idx], "local_target") or self.pop[idx].local_target is None:
            self.pop[idx].local_target = fallback_target.copy()
            self.pop[idx].local_solution = self.pop[idx].solution.copy()

    def evolve(self, epoch):
        w = (
            self.w_max
            - (self.w_max - self.w_min)
            * epoch
            / self.epoch
        )

        pop_new = []

        for idx in range(self.pop_size):
            pos = self.pop[idx].solution.copy()
            vel = self.pop[idx].velocity.copy()
            exemplar = self.fi[idx]

            new_vel = np.zeros(self.problem.n_dims)
            for d in range(self.problem.n_dims):
                exemplar_idx = exemplar[d]
                pbest_d = self.pop[exemplar_idx].local_solution[d]
                r = self.generator.random()
                new_vel[d] = w * vel[d] + self.c * r * (pbest_d - pos[d])

            new_vel = np.clip(new_vel, self.v_min, self.v_max)
            new_pos = pos + new_vel
            new_pos = self.correct_solution(new_pos)

            agent = self.generate_empty_agent(new_pos)
            agent.velocity = new_vel.copy()
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(new_pos)

                self.pop[idx] = self.get_better_agent(
                    self.pop[idx], agent, self.problem.minmax
                )
                # Fix: ensure local_target exists after greedy selection
                self._ensure_local_target(idx, agent.target)

                if self.compare_target(agent.target, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].update(
                        local_solution=agent.solution.copy(),
                        local_target=agent.target.copy()
                    )
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1

                if self.flags[idx] >= self.max_flag:
                    self.refresh_exemplars(idx)
                    self.flags[idx] = 0

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

            for idx in range(self.pop_size):
                pop_new[idx].velocity = pop_new[idx].velocity.copy()

                self.pop[idx] = self.get_better_agent(
                    self.pop[idx], pop_new[idx], self.problem.minmax
                )
                # Fix: ensure local_target exists after greedy selection
                self._ensure_local_target(idx, pop_new[idx].target)

                if self.compare_target(pop_new[idx].target, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].update(
                        local_solution=pop_new[idx].solution.copy(),
                        local_target=pop_new[idx].target.copy()
                    )
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1

                if self.flags[idx] >= self.max_flag:
                    self.refresh_exemplars(idx)
                    self.flags[idx] = 0
from mealpy.optimizer import Optimizer
from mealpy.swarm_based.GWO import OriginalGWO, WS_GWO, CL_DSel_GWO, CL_GWO, RLCGWO, AMGWO_GenliangLi, RLCGWO_ZhengpengHu, BFLGWO_YongWang
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.DE import OriginalDE


class _CLGWOAblationBase(Optimizer):
    """
    Shared scaffolding for the CL-GWO ablation variants. Reproduces CL_GWO's
    evolve() from mealpy.swarm_based.GWO (including the max_flag stagnation
    counter, kept for parameter parity even though it is bookkeeping-only and
    does not alter the search), with the per-dimension switch and the
    competitive (tournament) leader selection each individually toggleable so
    their contributions can be isolated (Reviewer 2, point 5). All constructor
    defaults match mealpy's CL_GWO (w_min=0.2, w_max=2.0, max_flag=7) so that
    the only variable differing between CL_GWO, CL_GWO_NoDWL and CL_GWO_NoALS
    is the mechanism removed, not a hyperparameter.
    """

    def __init__(self, epoch=10000, pop_size=100, w_min=0.2, w_max=2.0,
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

    def _select_leader(self, idx, alpha, per_dimension):
        """Return the leader agent for this decision, honoring per-dimension switching."""
        raise NotImplementedError

    def evolve(self, epoch):
        a = self.w_max - (self.w_max - self.w_min) * (epoch / self.epoch)

        _, leaders, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        alpha, beta, delta = leaders

        pop_new = []

        for idx in range(self.pop_size):
            X_new = self.pop[idx].solution.copy()
            pci = 0.05 + 0.45 * idx / self.pop_size

            agent_leader = self._pick_agent_level_leader(idx, alpha, pci)

            for d in range(self.problem.n_dims):
                leader = self._pick_dimension_leader(idx, alpha, pci, agent_leader)

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

                # Stagnation counter, kept identical to CL_GWO for parameter
                # parity. Bookkeeping only: no corrective action is triggered.
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

    def _pick_agent_level_leader(self, idx, alpha, pci):
        """Only used by variants that decide the leader once per agent, not per dimension."""
        return None

    def _pick_dimension_leader(self, idx, alpha, pci, agent_leader):
        raise NotImplementedError

    def _competitive_peer(self, idx):
        """Tournament: draw two peers, return the one with the better target (fitness)."""
        id1, id2 = self.generator.choice(
            list(set(range(self.pop_size)) - {idx}), 2, replace=False
        )
        return self.pop[id1] if self.compare_target(
            self.pop[id1].target, self.pop[id2].target, self.problem.minmax
        ) else self.pop[id2]

    def _random_peer(self, idx):
        """No tournament: a single uniformly random peer."""
        peer_id = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
        return self.pop[peer_id]


class CL_GWO_NoALS(_CLGWOAblationBase):
    """
    Ablation: dimension-wise learning (DWL) ON, Adaptive Leader Selection
    (ALS) OFF. Each dimension independently decides exploit-vs-explore,
    but the exploration peer is drawn uniformly at random instead of via
    tournament comparison.
    """

    def _pick_dimension_leader(self, idx, alpha, pci, agent_leader):
        if self.generator.random() > pci:
            return alpha
        return self._random_peer(idx)


class CL_GWO_NoDWL(_CLGWOAblationBase):
    """
    Ablation: Adaptive Leader Selection (ALS) ON, dimension-wise learning
    (DWL) OFF. A single exploit-vs-explore decision (with tournament
    selection when exploring) is made once per agent and applied uniformly
    across all dimensions.
    """

    def _pick_agent_level_leader(self, idx, alpha, pci):
        if self.generator.random() > pci:
            return alpha
        return self._competitive_peer(idx)

    def _pick_dimension_leader(self, idx, alpha, pci, agent_leader):
        return agent_leader


def create_optimizer(cfg, callback=None):

    algo = cfg["META_ALGO"]
    pop  = cfg["POP_SIZE"]
    it   = cfg["MAX_ITER"]
    seed = cfg.get("RANDOM_STATE")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if algo == "PSO":
        return OriginalPSO(epoch=it, pop_size=pop, c1=1.5, c2=1.5, w=0.7, callback=callback)
    if algo == "AIWPSO":
        return AIW_PSO(epoch=it, pop_size=pop, c1=1.5, c2=1.5, w=0.7, callback=callback)
    if algo == "LDW_PSO":
        return LDW_PSO(epoch=it, pop_size=pop, c1=1.5, c2=1.5, w=0.7, callback=callback)
    if algo == "CLPSO_Liang":
        return PatchedCLPSO_Liang(epoch=it, pop_size=pop, c=1.49445, w_max=0.9, w_min=0.4, max_flag=7, callback=callback)
    if algo == "CLPSO_RAP":
        return CLPSO_RAP(epoch=it, pop_size=pop, c=1.49445, w_max=0.9, w_min=0.4, max_flag=7, callback=callback)
    if algo == "GWO":
        return OriginalGWO(epoch=it, pop_size=pop, callback=callback)
    if algo == "GA":
        return BaseGA(epoch=it, pop_size=pop, pc=0.9, pm=0.1, callback=callback)
    if algo == "DE":
        return OriginalDE(epoch=it, pop_size=pop, wf=0.8, cr=0.9, callback=callback)
    if algo == "WS_GWO":
        return WS_GWO(epoch=it, pop_size=pop, callback=callback)
    if algo == "RLCGWO":
        return RLCGWO(epoch=it, pop_size=pop, callback=callback)
    if algo == "AMGWO_GenliangLi":
        return AMGWO_GenliangLi(epoch=it, pop_size=pop, callback=callback)
    if algo == "RLCGWO_ZhengpengHu":
        return RLCGWO_ZhengpengHu(epoch=it, pop_size=pop, callback=callback)
    if algo == "BFLGWO_YongWang":
        return BFLGWO_YongWang(epoch=it, pop_size=pop, callback=callback)
    if algo == "CL_DSel_GWO":
        return CL_DSel_GWO(epoch=it, pop_size=pop, callback=callback)
    
    # w_min=0.2, w_max=2.0, max_flag=7 are mealpy's CL_GWO defaults and match
    # the values reported in the manuscript (Section 3.6 / Table 4). They are
    # passed explicitly here (instead of relying on defaults) so CL_GWO,
    # CL_GWO_NoDWL and CL_GWO_NoALS are guaranteed to run with identical
    # hyperparameters for the ablation study.
    if algo == "CL_GWO":
        return CL_GWO(epoch=it, pop_size=pop, w_min=0.2, w_max=2.0, max_flag=7, callback=callback)
    if algo == "CL_GWO_NoDWL":
        return CL_GWO_NoDWL(epoch=it, pop_size=pop, w_min=0.2, w_max=2.0, max_flag=7, callback=callback)
    if algo == "CL_GWO_NoALS":
        return CL_GWO_NoALS(epoch=it, pop_size=pop, w_min=0.2, w_max=2.0, max_flag=7, callback=callback)

    raise ValueError(f"Unknown META_ALGO: {algo}")
