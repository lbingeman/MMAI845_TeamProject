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

```bash
python main.py <command> <network_type> <environment_mode> [continue]
```

#### Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `command`        | What to do: `train`, `run`, or `evaluate`                                   |
| `network_type`   | Model to use: `q_learning`, `dqn`, `sarsa`, `stable_baseline_dqn`           |
| `environment_mode` | Mode/environment (e.g., `"train"`, `"test"`, etc. – based on your setup) |
| `continue`       | *(Optional)* Use `"continue"` to resume training from a saved model         |

---

### ✅ Examples

**Train a DQN model from scratch**  
```bash
python main.py train dqn train
```

**Continue training a Q-Learning model**  
```bash
python main.py train q_learning train continue
```

**Evaluate a saved SARSA model**  
```bash
python main.py evaluate sarsa eval
```

**Run simulation using a Stable Baseline DQN model**  
```bash
python main.py run stable_baseline_dqn test
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
