from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modeling import RiskModel


@dataclass
class PolicyConfig:
    preventive_cost: float = 18.0
    failure_cost: float = 95.0
    horizon: int = 4000


@dataclass
class DQNConfig:
    hidden_dim: int = 24
    learning_rate: float = 0.008
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    target_sync_every: int = 50


class DQNPolicy:
    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        horizon: float,
    ) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.horizon = horizon

    def _obs_to_vector(self, obs: dict[str, float]) -> np.ndarray:
        risk = float(obs["risk"])
        age_scaled = float(obs["age"]) / self.horizon
        risk_delta = float(obs["risk_delta"])
        return np.array([risk, age_scaled, risk_delta, risk * age_scaled, risk**2], dtype=float)

    def q_values(self, obs: dict[str, float]) -> np.ndarray:
        x = self._obs_to_vector(obs)[None, :]
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2)[0]

    def __call__(self, state: tuple[int, int, int], obs: dict[str, float]) -> int:
        del state
        return int(np.argmax(self.q_values(obs)))


RISK_EDGES = [0.10, 0.25, 0.45, 0.65]
AGE_EDGES = [25, 50, 100, 150, 200]
RISK_DELTA_EDGES = [-0.02, 0.02, 0.08]


class MaintenanceEnv:
    def __init__(
        self,
        seed: int,
        risk_model: RiskModel,
        trajectory_pool: list[pd.DataFrame],
        policy_config: PolicyConfig | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.risk_model = risk_model
        self.trajectory_pool = trajectory_pool
        self.policy_config = policy_config or PolicyConfig()
        self.machine_df = pd.DataFrame()
        self.idx = 0
        self.total_time = 0
        self.done = False
        self.n_replacements = 0
        self.n_failures = 0

    def reset(self) -> tuple[tuple[int, int, int], dict[str, float]]:
        self.total_time = 0
        self.done = False
        self.n_replacements = 0
        self.n_failures = 0
        self._load_new_machine()
        obs = self._current_obs()
        return self._state(obs), obs

    def _load_new_machine(self) -> None:
        selected = int(self.rng.integers(0, len(self.trajectory_pool)))
        self.machine_df = self.trajectory_pool[selected].reset_index(drop=True).copy()
        self.idx = 0

    def _current_obs(self) -> dict[str, float]:
        row = self.machine_df.iloc[self.idx]
        prev_risk = float(self.machine_df.iloc[self.idx - 1]["predicted_risk"]) if self.idx > 0 else 0.0
        current_risk = float(row["predicted_risk"])
        return {
            "risk": current_risk,
            "risk_delta": current_risk - prev_risk,
            "age": float(row["age"]),
            "failed": float(row["failed"]),
        }

    @staticmethod
    def _risk_bucket(risk: float) -> int:
        return int(np.digitize([risk], RISK_EDGES)[0])

    @staticmethod
    def _age_bucket(age: float) -> int:
        return int(np.digitize([age], AGE_EDGES)[0])

    @staticmethod
    def _risk_delta_bucket(risk_delta: float) -> int:
        return int(np.digitize([risk_delta], RISK_DELTA_EDGES)[0])

    def _state(self, obs: dict[str, float]) -> tuple[int, int, int]:
        return (
            self._risk_bucket(obs["risk"]),
            self._age_bucket(obs["age"]),
            self._risk_delta_bucket(obs["risk_delta"]),
        )

    def step(self, action: int) -> tuple[tuple[int, int, int], dict[str, float], float, bool]:
        cost = 0.0
        if action == 1:
            cost += self.policy_config.preventive_cost
            self.n_replacements += 1
            self._load_new_machine()
        else:
            current = self.machine_df.iloc[self.idx]
            if int(current["failed"]) == 1:
                cost += self.policy_config.failure_cost
                self.n_failures += 1
                self.n_replacements += 1
                self._load_new_machine()
            else:
                self.idx += 1
                if self.idx >= len(self.machine_df):
                    cost += self.policy_config.failure_cost
                    self.n_failures += 1
                    self.n_replacements += 1
                    self._load_new_machine()

        self.total_time += 1
        self.done = self.total_time >= self.policy_config.horizon
        obs = self._current_obs()
        return self._state(obs), obs, -cost, self.done


def evaluate_policy(
    env_seed: int,
    risk_model: RiskModel,
    trajectory_pool: list[pd.DataFrame],
    policy_name: str,
    policy_fn,
    n_episodes: int = 14,
) -> dict[str, float | str]:
    rewards = []
    replacements = []
    failures = []
    for episode in range(n_episodes):
        env = MaintenanceEnv(seed=env_seed + episode, risk_model=risk_model, trajectory_pool=trajectory_pool)
        state, obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy_fn(state, obs)
            state, obs, reward, done = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        replacements.append(env.n_replacements)
        failures.append(env.n_failures)
    return {
        "policy": policy_name,
        "avg_reward": float(np.mean(rewards)),
        "avg_cost": float(-np.mean(rewards)),
        "std_cost": float(np.std([-reward for reward in rewards], ddof=0)),
        "avg_replacements": float(np.mean(replacements)),
        "avg_failures": float(np.mean(failures)),
    }


def threshold_sweep(
    valid_df: pd.DataFrame,
    risk_model: RiskModel,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    preds = risk_model.predict(valid_df)
    y = valid_df["fail_within_horizon"].to_numpy()
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preventive = preds >= threshold
        expected_cost = np.mean(preventive * 18.0 + (~preventive) * y * 95.0)
        preventive_rate = float(np.mean(preventive))
        miss_rate = float(np.mean((~preventive) * y))
        rows.append(
            {
                "threshold": float(threshold),
                "expected_cost": float(expected_cost),
                "preventive_rate": preventive_rate,
                "miss_rate": miss_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def tune_threshold_policy(valid_df: pd.DataFrame, risk_model: RiskModel) -> float:
    sweep_df = threshold_sweep(valid_df, risk_model)
    best_row = sweep_df.sort_values(["expected_cost", "threshold"], ascending=[True, True]).iloc[0]
    return float(best_row["threshold"])


def train_q_learning_policy(
    risk_model: RiskModel,
    trajectory_pool: list[pd.DataFrame],
    seed: int = 2026,
    n_episodes: int = 120,
    alpha: float = 0.12,
    gamma: float = 0.98,
    epsilon: float = 0.16,
) -> np.ndarray:
    q = np.zeros((len(RISK_EDGES) + 1, len(AGE_EDGES) + 1, len(RISK_DELTA_EDGES) + 1, 2), dtype=float)
    rng = np.random.default_rng(seed)
    for episode in range(n_episodes):
        env = MaintenanceEnv(seed=seed + episode, risk_model=risk_model, trajectory_pool=trajectory_pool)
        state, obs = env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(q[state[0], state[1], state[2]]))
            next_state, obs, reward, done = env.step(action)
            shaped_reward = reward - (4.0 * float(obs["risk"]) if action == 0 else 0.0)
            td_target = shaped_reward + (0.0 if done else gamma * np.max(q[next_state[0], next_state[1], next_state[2]]))
            td_error = td_target - q[state[0], state[1], state[2], action]
            q[state[0], state[1], state[2], action] += alpha * td_error
            state = next_state
    return q


def q_policy_from_table(q_table: np.ndarray):
    def _policy(state: tuple[int, int, int], obs: dict[str, float]) -> int:
        del obs
        return int(np.argmax(q_table[state[0], state[1], state[2]]))

    return _policy


def train_dqn_policy(
    risk_model: RiskModel,
    trajectory_pool: list[pd.DataFrame],
    seed: int = 2026,
    n_episodes: int = 24,
    config: DQNConfig | None = None,
) -> tuple[DQNPolicy, pd.DataFrame]:
    config = config or DQNConfig()
    rng = np.random.default_rng(seed)
    obs_dim = 5
    action_dim = 2
    horizon = float(PolicyConfig().horizon)

    w1 = rng.normal(0.0, 0.25, size=(obs_dim, config.hidden_dim))
    b1 = np.zeros(config.hidden_dim, dtype=float)
    w2 = rng.normal(0.0, 0.25, size=(config.hidden_dim, action_dim))
    b2 = np.zeros(action_dim, dtype=float)

    target_w1 = w1.copy()
    target_b1 = b1.copy()
    target_w2 = w2.copy()
    target_b2 = b2.copy()

    total_steps = 0
    epsilon = config.epsilon_start
    history_rows: list[dict[str, float | int]] = []

    def obs_to_vector(obs: dict[str, float]) -> np.ndarray:
        risk = float(obs["risk"])
        age_scaled = float(obs["age"]) / horizon
        risk_delta = float(obs["risk_delta"])
        return np.array([risk, age_scaled, risk_delta, risk * age_scaled, risk**2], dtype=float)

    def forward(
        x: np.ndarray,
        weight_1: np.ndarray,
        bias_1: np.ndarray,
        weight_2: np.ndarray,
        bias_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden_linear = x @ weight_1 + bias_1
        hidden = np.maximum(0.0, hidden_linear)
        q_values = hidden @ weight_2 + bias_2
        return hidden, q_values

    def train_step(state_vec: np.ndarray, action: int, reward: float, next_state_vec: np.ndarray, done: bool) -> float:
        nonlocal w1, b1, w2, b2
        hidden, q_values = forward(state_vec[None, :], w1, b1, w2, b2)
        _, next_q_target = forward(next_state_vec[None, :], target_w1, target_b1, target_w2, target_b2)
        target = reward + (0.0 if done else config.gamma * float(np.max(next_q_target[0])))
        td_error = float(q_values[0, action] - target)
        loss = td_error**2

        grad_q = np.zeros_like(q_values)
        grad_q[0, action] = 2.0 * td_error

        grad_w2 = hidden.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)

        grad_hidden = grad_q @ w2.T
        grad_hidden[hidden <= 0.0] = 0.0
        grad_w1 = state_vec[:, None] @ grad_hidden
        grad_b1 = grad_hidden[0]

        w2 -= config.learning_rate * grad_w2
        b2 -= config.learning_rate * grad_b2
        w1 -= config.learning_rate * grad_w1
        b1 -= config.learning_rate * grad_b1
        return float(loss)

    for episode in range(n_episodes):
        env = MaintenanceEnv(seed=seed + episode, risk_model=risk_model, trajectory_pool=trajectory_pool)
        _, obs = env.reset()
        done = False
        total_reward = 0.0
        episode_losses: list[float] = []

        while not done:
            state_vec = obs_to_vector(obs)
            if rng.random() < epsilon:
                action = int(rng.integers(0, action_dim))
            else:
                _, q_values = forward(state_vec[None, :], w1, b1, w2, b2)
                action = int(np.argmax(q_values[0]))

            _, next_obs, reward, done = env.step(action)
            shaped_reward = reward - (4.0 * float(next_obs["risk"]) if action == 0 else 0.0)
            next_state_vec = obs_to_vector(next_obs)
            total_reward += reward
            obs = next_obs
            total_steps += 1
            episode_losses.append(train_step(state_vec, action, shaped_reward, next_state_vec, done))

            if total_steps % config.target_sync_every == 0:
                target_w1 = w1.copy()
                target_b1 = b1.copy()
                target_w2 = w2.copy()
                target_b2 = b2.copy()

        history_rows.append(
            {
                "episode": episode + 1,
                "reward": float(total_reward),
                "epsilon": float(epsilon),
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else np.nan,
            }
        )
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    policy = DQNPolicy(w1=w1, b1=b1, w2=w2, b2=b2, horizon=horizon)
    return policy, pd.DataFrame(history_rows)
