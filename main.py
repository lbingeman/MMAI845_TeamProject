import sys
from FarmDeliveryRLSystem import FarmDeliveryRLSystemBase
from FarmDeliveryDQN import FarmDQNModel
from FarmingDeliveryQLearning import FarmQLearningModel
from FarmDeliveryStableBaseline import FarmStableBaselineDQNModel
from FarmDeliverySarsaModel import FarmSarsaModel
import pickle

def get_standard_model(network_type, environment_mode, should_load) -> FarmDeliveryRLSystemBase:
    model = None
    if network_type == "q_learning":
        model = FarmQLearningModel(environment_mode=environment_mode, use_action_mask=True)
    elif network_type == "dqn":
        model = FarmDQNModel(environment_mode=environment_mode)
    elif network_type == "sarsa":
        model = FarmSarsaModel(environment_mode=environment_mode, use_action_mask=True)
    
    if should_load:
        with open(model.filename, "rb") as f:
                return pickle.load(f)
    else:
        return model

def get_model(network_type, environment_mode, should_load) -> FarmDeliveryRLSystemBase:
    model = None
    if network_type == "q_learning" or network_type == "dqn" or network_type == "sarsa":
        return get_standard_model(network_type, environment_mode, should_load)
    elif network_type == "stable_baseline_dqn":
        model = FarmStableBaselineDQNModel(environment_mode=environment_mode)
        
        if should_load:
            model.reload_model()
            
        return model

def get_should_load():
    if command == "run":
        return True
    elif command == "train":
        if len(sys.argv) >= 5:
            return sys.argv[4] == "continue"
        return False
    elif command == "evaluate":
        return True

if __name__=='__main__':
    if len(sys.argv) >= 4:
        command = sys.argv[1]
        network_type = sys.argv[2]
        environment_mode = sys.argv[3]
        
        should_load = get_should_load()
        model = get_model(network_type, environment_mode, should_load)
        
        if command == "run":
            model.run() 
        elif command == "train":
            model.train() 
        elif command == "evaluate":
            model.evaluate()