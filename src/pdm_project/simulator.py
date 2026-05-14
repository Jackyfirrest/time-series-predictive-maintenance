from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulatorConfig:
    max_steps: int = 180
    failure_threshold: float = 0.14


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_machine(machine_id: int, rng: np.random.Generator, config: SimulatorConfig) -> pd.DataFrame:
    health = 1.0
    regime = 0
    base_deg = rng.uniform(0.0035, 0.0065)
    stress_bonus = rng.uniform(0.0060, 0.0130)
    shock_scale = rng.uniform(0.0, 0.012)
    rows = []

    for t in range(config.max_steps):
        if regime == 0:
            regime = int(rng.random() < 0.06)
        else:
            regime = int(rng.random() < 0.82)

        shock = shock_scale if rng.random() < 0.05 else 0.0
        damage = base_deg + stress_bonus * regime + shock + rng.normal(0.0, 0.0012)
        damage = max(damage, 0.0005)
        health = max(health - damage, -0.05)

        vibration = 0.40 + 1.65 * (1.0 - health) + 0.22 * regime + 0.08 * np.sin(t / 5.0) + rng.normal(0.0, 0.05)
        temperature = 60.0 + 20.0 * (1.0 - health) + 4.5 * regime + rng.normal(0.0, 0.85)
        pressure = 105.0 - 11.0 * (1.0 - health) - 1.2 * regime + rng.normal(0.0, 0.7)

        hazard = _sigmoid(-8.3 + 13.0 * (1.0 - health) + 0.7 * regime)
        failed = int(health <= config.failure_threshold or rng.random() < hazard)

        rows.append(
            {
                "machine_id": machine_id,
                "t": t,
                "age": t + 1,
                "regime": regime,
                "health": health,
                "vibration": vibration,
                "temperature": temperature,
                "pressure": pressure,
                "failed": failed,
            }
        )
        if failed:
            break

    df = pd.DataFrame(rows)
    failure_time = int(df.loc[df["failed"] == 1, "t"].iloc[0])
    df["failure_time"] = failure_time
    df["time_to_failure"] = failure_time - df["t"]
    return df


def simulate_fleet(n_machines: int, seed: int, config: SimulatorConfig | None = None) -> pd.DataFrame:
    config = config or SimulatorConfig()
    rng = np.random.default_rng(seed)
    frames = [simulate_machine(machine_id=i, rng=rng, config=config) for i in range(n_machines)]
    return pd.concat(frames, ignore_index=True)

