<!--
 * @Author: WANG Maonan
 * @Date: 2026-01-10 17:05:12
 * @Description: Single-intersection TSC baselines built on TransSimHub + SUMO
 * @LastEditTime: 2026-06-25 20:03:05
-->
# Single-Intersection Traffic Signal Control Baselines

A standardized benchmark for **single-intersection Traffic Signal Control (TSC)**, built on [TransSimHub (tshub)](https://github.com/Traffic-Alpha/TransSimHub) and the [SUMO](https://www.eclipse.org/sumo/) simulator. It provides a common environment layer and a set of comparable traditional and reinforcement-learning controllers, evaluated across a wide range of real-world intersections and traffic patterns.

## Features

- **Unified environment layer** (`tsc_env/`) shared by every algorithm — static/dynamic feature extraction, pluggable rewards/observations, and configurable action types.
- **Traditional baselines**: Fixed-Time, Max Pressure, Webster, SOTL.
- **RL baselines**: PressLight, AttendLight, IntelliLight, UniTSA (Stable-Baselines3).
- **12 real-world intersection scenarios**, each with two network complexities × five traffic-demand patterns.
- **Special-event injection** for robustness testing — inject accidents (static barriers) and special vehicles (ambulance / police / fire) at evaluation time.

## Architecture

`tsc_env/` provides the building blocks; each algorithm assembles its own pipeline through its own `make_env.py`.

```
                                     ┌─ (eval only) ─┐
SUMO ─► TshubEnvironment ─► TSCEnvironment ─► [TSCEventWrapper] ─► TSCInfoWrapper ─► …
                              (base_env.py)    (event_wrapper.py)   (tsc_info_wrapper.py)

Traditional:  … ─► TSCInfoWrapper ─► Agent reads dict features directly
RL:           … ─► TSCInfoWrapper ─► <algo>RLWrapper (reward/obs/action) ─► Monitor (SB3)
```

- **`TSCEnvironment`** (`tsc_env/base_env.py`) — `gym.Env` wrapper around SUMO/tshub.
- **`TSCInfoWrapper`** (`tsc_env/tsc_info_wrapper.py`) — extracts static + dynamic lane/TLS features.
- **`TSCEventWrapper`** (`tsc_env/event_wrapper.py`) — optional, injects accidents and special vehicles.
- Each RL algorithm adds its own RL wrapper (reward function, observation builder, action type).

## Repository Layout

```
tsc_env/                      # Shared environment building blocks
  base_env.py                 #   TSCEnvironment: SUMO/tshub interface
  tsc_info_wrapper.py         #   Static + dynamic feature extraction
  event_wrapper.py            #   Special-event injection (accidents / special vehicles)
  tools/                      #   Cell / static / TLS feature helpers
  tsc_visualizer.py           #   Lane-feature & congestion visualization

tsc_algos/
  traditional/                # Rule-based controllers (fixtime, maxpressure, webster, sotl)
    base_traditional.py       #   Base class with the run() loop
    <algo>/{make_env,run}.py  #   Per-algo pipeline + entry point
  rl/                         # SB3-based RL controllers
    presslight/  attendlight/  intellilight/  unitsa/
      <algo>_env/             #   make_env, reward_funcs, rl_wrapper, state_funcs
      model.py                #   Network (MLP / movement-token Transformer)
      train.py  eval.py       #   Entry points
    utils/                    #   Shared SB3 utilities

junction_configs/             # Per-junction config (env params + EVENTS) + loader
junction_scenarios/           # SUMO scenarios (networks, routes, detectors, configs)
assets/                       # Figures and traffic-flow analysis helpers
```

## Algorithms

### Traditional

| Algorithm    | Description                          | Entry point                                |
|--------------|--------------------------------------|--------------------------------------------|
| FixTime      | Fixed-time phase cycling             | `tsc_algos/traditional/fixtime/run.py`     |
| MaxPressure  | Max-pressure phase selection         | `tsc_algos/traditional/maxpressure/run.py` |
| Webster      | Webster's cycle/split method         | `tsc_algos/traditional/webster/run.py`     |
| SOTL         | Self-Organizing Traffic Lights       | `tsc_algos/traditional/sotl/run.py`        |

### Reinforcement Learning

| Algorithm    | RL algo | Action            | Reward                                    | Network                       |
|--------------|---------|-------------------|-------------------------------------------|-------------------------------|
| PressLight   | DQN     | choose_next_phase | movement pressure                         | MLP                           |
| AttendLight  | DQN     | choose_next_phase | movement pressure                         | movement-token Transformer    |
| IntelliLight | DQN     | choose_next_phase | average waiting time                      | MLP (+ VecNormalize)          |
| UniTSA       | PPO     | next_or_not       | cumulative waiting time (anti-starvation) | independent feature extractor |

## Installation

Requires **SUMO** (installed separately) and the `tshub` conda environment.

```bash
conda activate tshub      # or: conda run -n tshub <cmd>
```

Python dependencies: `tshub`, `stable_baselines3`, `gymnasium`, `torch`, `numpy`, `matplotlib`, `loguru`, `pyyaml`.

> All training / evaluation / SUMO scripts must run inside the `tshub` environment (not base).

## Quick Start

All entry points share `--junction` and `--env_name`. `env_name` has the form `{difficulty}_{pattern}`, e.g. `normal_low_density`.

### Traditional

```bash
python tsc_algos/traditional/maxpressure/run.py \
    --junction Beijing_Beihuan --env_name normal_low_density --use_gui
```

### RL — train & evaluate

```bash
# Train
python tsc_algos/rl/presslight/train.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter \
    --num_envs 8 --total_timesteps 300000

# Evaluate (writes SUMO tripinfo for metric comparison)
python tsc_algos/rl/presslight/eval.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter --gui
```

## Scenarios

`junction_scenarios/` contains **12 real-world intersections**:

```
Beijing_Beihuan        Beijing_Beishahe       Beijing_Changjianglu
Beijing_Gaojiaoyuan    Beijing_Pinganli       Beijing_Yongrunlu
Chengdu_Chenghannanlu  Chengdu_Guanghua       France_Massy
Hongkong_YMT           SouthKorea_Songdo      Tianjin_zhijingdao
```

Each junction provides two network complexities × five demand patterns (10 SUMO configs):

| Network            | Traffic patterns                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------------|
| `easy` / `normal`  | `low_density`, `high_density`, `fluctuating_commuter`, `increasing_demand`, `random_perturbation` |

A scenario directory holds `networks/`, `routes/`, `add/` (lane-area detectors), `generate_routes.py`, and the `{easy,normal}_{pattern}.sumocfg` files.

## Special Events (Robustness Evaluation)

Events are declared per junction in an `EVENTS` dict inside `junction_configs/<junction>.py`, selected at evaluation time with `--event_name`:

```python
# junction_configs/Beijing_Beihuan.py
EVENTS = {
    "event_1": {
        "accidents": [          # static barrier on a lane for `duration` seconds
            {"id": "accident_01", "depart_time": 60, "edge_id": "...",
             "lane_index": 1, "position": 218.5, "type": "barrier", "duration": 101},
        ],
        "special_vehicles": [   # ambulance / police / fire dispatched along a route
            {"id": "ambulance_02", "type": "ambulance", "depart_time": 100,
             "route": ["...", "..."]},
        ],
    },
}
```

```bash
# Works for every RL eval.py and traditional run.py
python tsc_algos/rl/unitsa/eval.py \
    --junction Beijing_Beihuan --env_name normal_low_density --event_name event_1 --gui
```

Omitting `--event_name` runs without events; the wrapper is only mounted when an event set is given. Vehicle `type`s must be defined in the scenario's route files — otherwise `TSCEventWrapper` copies `DEFAULT_VEHTYPE` as a fallback.

## Extending the Benchmark

- **New traditional algorithm** — create `tsc_algos/traditional/<name>/`, implement `choose_action()` (subclass `BaseTraditionalAgent`), and add `make_env.py` + `run.py`.
- **New RL algorithm** — create `tsc_algos/rl/<name>/` with a `<name>_env/` package (`make_env`, `reward_funcs`, `rl_wrapper`, `state_funcs`), `model.py`, `train.py`, `eval.py`.
- **New junction** — add a directory under `junction_scenarios/` (`networks/`, `routes/`, `add/`, `generate_routes.py`, SUMO configs) and a matching `junction_configs/<junction>.py`.

## Citation

If you find this work helpful, please consider citing the papers below.

### LLM / VLM-based TSC

```
@article{wang2024llm,
  title={LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments},
  author={Wang, Maonan and Pang, Aoyu and Kan, Yuheng and Pun, Man-On and Chen, Chung Shue and Huang, Bo},
  journal={arXiv preprint arXiv:2403.08337},
  year={2024}
}

@inproceedings{wang2025vlmlight,
 author = {Wang, Maonan and Chen, Yirong and Pang, Aoyu and Cai, Yuxin and Chen, Chung Shue and Kan, Yuheng and Pun, Man On},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {39590--39621},
 publisher = {Curran Associates, Inc.},
 title = {{VLMLight}: Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning Architecture},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/3849b5861dcaeaf4758eef0979a98cc6-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}

@ARTICLE{pang2026illm,
  author={Pang, Aoyu and Wang, Maonan and Pun, Man-On and Chen, Chung Shue and Xiong, Xi},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={{iLLM-TSC}: Integration Reinforcement Learning and Large Language Model for Traffic Signal Control Policy Improvement}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TVT.2026.3674284}
}
```

### RL-based TSC

```
@ARTICLE{wang2024unitsa,
  author={Wang, Maonan and Xiong, Xi and Kan, Yuheng and Xu, Chengcheng and Pun, Man-On},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={UniTSA: A Universal Reinforcement Learning Framework for V2X Traffic Signal Control}, 
  year={2024},
  volume={73},
  number={10},
  pages={14354-14369},
  doi={10.1109/TVT.2024.3403879}
}

@ARTICLE{wang2024ccda,
  author={Wang, Maonan and Chen, Yirong and Kan, Yuheng and Xu, Chengcheng and Lepech, Michael and Pun, Man-On and Xiong, Xi},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Traffic Signal Cycle Control With Centralized Critic and Decentralized Actors Under Varying Intervention Frequencies}, 
  year={2024},
  volume={25},
  number={12},
  pages={20085-20104},
  doi={10.1109/TITS.2024.3462153}
}

@ARTICLE{10443835,
  author={Pang, Aoyu and Wang, Maonan and Chen, Yirong and Pun, Man-On and Lepech, Michael},
  journal={IEEE Open Journal of Vehicular Technology}, 
  title={Scalable Reinforcement Learning Framework for Traffic Signal Control Under Communication Delays}, 
  year={2024},
  volume={5},
  number={},
  pages={330-343},
  doi={10.1109/OJVT.2024.3368693}
}
```

## License & Author

- **Author**: WANG Maonan
- Built on [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub).
