# Lane Change Decision Making using Reinforcement Learning in CARLA

A research-oriented implementation of lane change decision making using
Reinforcement Learning in the CARLA autonomous driving simulator.

---

## Features

- CARLA 0.9.15 support
- Gymnasium compatible environment
- High-level lane change decision making
- Classical low-level controller (PID)
- DQN and PPO baselines
- TensorBoard logging
- Evaluation framework
- Modular architecture

---

## Project Structure

```
lane-change-rl/
│
├── configs/
│   ├── config.yaml
│   └── loader.py
│
├── lane_change_rl/
│   ├── env/
│   ├── controllers/
│   ├── sensors/
│   ├── agents/
│   └── utils/
│
├── scripts/
│
├── tests/
│
├── logs/
│
├── pyproject.toml
└── README.md
```

---

## Requirements

- Ubuntu 22.04 (recommended)
- Python 3.10+
- CARLA 0.9.15

---

## Installation

Clone the repository.

```bash
git clone <repo-url>
cd lane-change-rl
```

Create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package.

```bash
pip install -e .
```

Install the CARLA Python API matching your simulator version.

Example:

```bash
pip install carla==0.9.15
```

---

## Running CARLA

Start the simulator.

```bash
cd ~/CARLA_0.9.15
./CarlaUE4.sh
```

---

## Running the World Test

```bash
python scripts/test_world.py
```

---

## Planned Roadmap

### Phase 1

- World manager
- Actor manager
- Traffic manager
- Map manager

### Phase 2

- Sensors
- Observation extraction
- Gymnasium environment

### Phase 3

- Reward function
- Episode management

### Phase 4

- DQN training

### Phase 5

- PPO training

### Phase 6

- Evaluation and visualization

---

## Coding Style

- Black
- isort
- Type hints
- NumPy-style documentation

---

## License

MIT License