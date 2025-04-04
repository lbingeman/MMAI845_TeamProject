import sys
import pickle
from FarmDeliveryRLSystem import FarmDeliveryRLSystemBase
from FarmDeliveryDQN import FarmDQNModel
from FarmingDeliveryQLearning import FarmQLearningModel
from FarmDeliveryStableBaseline import FarmStableBaselineDQNModel
from FarmDeliverySarsaModel import FarmSarsaModel

def get_standard_model(network_type, environment_mode, should_load) -> FarmDeliveryRLSystemBase:
    """
    Returns a standard model instance based on the specified network type.
    
    Args:
        network_type (str): Type of the network ("q_learning", "dqn", or "sarsa").
        environment_mode (str): Environment mode (e.g., "train", "eval", etc.).
        should_load (bool): Whether to load a pre-trained model from disk.

    Returns:
        FarmDeliveryRLSystemBase: An instance of the appropriate model class.
    """
    model = None

    # Initialize the model based on the selected network type
    if network_type == "q_learning":
        model = FarmQLearningModel(environment_mode=environment_mode, use_action_mask=True)
    elif network_type == "dqn":
        model = FarmDQNModel(environment_mode=environment_mode)
    elif network_type == "sarsa":
        model = FarmSarsaModel(environment_mode=environment_mode, use_action_mask=True)

    # Load model from file if requested
    if should_load:
        with open(model.filename, "rb") as f:
            return pickle.load(f)
    else:
        return model

def get_model(network_type, environment_mode, should_load) -> FarmDeliveryRLSystemBase:
    """
    Returns the appropriate model instance depending on the specified network type.

    Args:
        network_type (str): Type of the network to use.
        environment_mode (str): The mode the environment should operate in.
        should_load (bool): Whether to load an existing model.

    Returns:
        FarmDeliveryRLSystemBase: Configured model instance.
    """
    model = None

    # Delegate standard models to the helper function
    if network_type in ["q_learning", "dqn", "sarsa"]:
        return get_standard_model(network_type, environment_mode, should_load)
    
    # Handle Stable Baseline DQN separately
    elif network_type == "stable_baseline_dqn":
        model = FarmStableBaselineDQNModel(environment_mode=environment_mode)
        
        if should_load:
            model.reload_model()

        return model

def get_should_load():
    """
    Determines whether a model should be loaded based on the current command and CLI arguments.

    Returns:
        bool: True if the model should be loaded, False otherwise.
    """
    if command == "run":
        return True
    elif command == "train":
        # Check for "continue" flag as the 5th argument to continue training
        if len(sys.argv) >= 5:
            return sys.argv[4] == "continue"
        return False
    elif command == "evaluate":
        return True

if __name__ == '__main__':
    # Check if sufficient CLI arguments are provided
    if len(sys.argv) >= 4:
        command = sys.argv[1]
        network_type = sys.argv[2]
        environment_mode = sys.argv[3]

        # Decide whether to load an existing model
        should_load = get_should_load()

        # Get the appropriate model
        model = get_model(network_type, environment_mode, should_load)

        # Perform the specified command
        if command == "run":
            model.run()
        elif command == "train":
            model.train()
        elif command == "evaluate":
            model.evaluate()
