# 🧑‍🌾 Farm Delivery RL System

This project provides a modular reinforcement learning framework for simulating and training agents in a farm delivery environment using various RL algorithms like Q-Learning, DQN, SARSA, and Stable Baselines DQN.

## 📦 Features

- Plug-and-play architecture for multiple RL model types.
- Supports loading from saved models.
- CLI-based training, evaluation, and simulation.
- Built-in model continuation support for training.

---

## 🚀 Usage

### Prerequisites

- Python 3.8+
- Required dependencies (based on your environment setup):

```bash
pip install -r requirements.txt
```

> Make sure your local environment includes the necessary model classes:
- `FarmDQNModel`
- `FarmQLearningModel`
- `FarmSarsaModel`
- `FarmStableBaselineDQNModel`

---

### 📁 File Structure

```
.
├── main.py
├── FarmDeliveryRLSystem.py
├── FarmDeliveryDQN.py
├── FarmingDeliveryQLearning.py
├── FarmDeliverySarsaModel.py
├── FarmDeliveryStableBaseline.py
└── ...
```

---

### 🔧 CLI Usage

#### Standard Model Operating
```bash
python main.py <command> <network_type> <environment_mode> [continue]
```

#### Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `command`        | What to do: `train`, `run`, or `evaluate`                                   |
| `network_type`   | Model to use: `q_learning`, `dqn`, `sarsa`, `stable_baseline_dqn`           |
| `environment_mode` | Mode/environment (e.g. `fixed_location`, `fully_fixed_state`, `fixed_orders`, `non_fixed_state` |
| `continue`       | *(Optional)* Use `"continue"` to resume training from a saved model         |

#### Warm Starting a DQN model
If you want to warm start your DQN model:
```bash
python main.py train dqn <environment_mode> warmstart <file_path>
```

#### Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `environment_mode` | Mode/environment (e.g. `fixed_location`, `fully_fixed_state`, `fixed_orders`, `non_fixed_state` |
| `file_path`       | File path to pkl file to load model weights from         |

---

#### Loading an existing model
```bash
python main.py <command> from <file_path>
```

#### Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `command`        | What to do: `train`, `run`, or `evaluate`                                   |
| `file_path`       | File path to pkl file containing saved model         |

### ✅ Examples

**Train a DQN model from scratch for fixed order environment**  
```bash
python main.py train dqn fixed_orders
```

**Continue training a Q-Learning model for fixed location environment**  
```bash
python main.py train q_learning fixed_location continue
```

**Evaluate a saved SARSA model with fixed location environment**  
```bash
python main.py evaluate sarsa fixed_location
```

**Run simulation using a Stable Baseline DQN model for fully fixed environment **  
```bash
python main.py run stable_baseline_dqn fully_fixed_state
```

**Evaluate a previous model**  
```bash
python main.py evaluate from dqn_model_delivery_non_fixed_state.pkl
```

---

### 💾 Model Persistence

- Models will automatically load from saved files if `should_load` is `True`.
- `FarmStableBaselineDQNModel` uses its own `reload_model()` method to handle loading.
- For standard models (DQN, Q-Learning, SARSA), the model is loaded via `pickle`.

---

## 📬 Contributing

Have an idea or improvement? Open a PR or start a discussion!

---

## 🧠 Author Notes

This system was designed for flexibility and experimentation with reinforcement learning in custom environments. The base interface is defined in `FarmDeliveryRLSystemBase`, and you can easily extend it with your own models.

