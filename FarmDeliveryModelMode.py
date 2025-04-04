from enum import Enum
from FarmingEnvironment import FarmEnv

class FarmDeliveryModelType(Enum):
    """
    Enum for different types of farming delivery models. 
    Each model corresponds to different ways of handling order and location variability.
    """
    # Location of each delivery request is fixed 
    FIXED_LOCATION = "fixed_location"
    # Order and Location of each delivery request is fixed 
    FULLY_FIXED_STATE = "fully_fixed_state"
    # Order of each delivery request is fixed
    FIXED_ORDERS = "fixed_orders"
    # Order and location of each delivery request is randomized
    NON_FIXED_STATE = "non_fixed_state"

    @classmethod
    def from_string(cls, model_type: str):
        """
        Converts a string to the corresponding FarmDeliveryModelType enum value.
        
        Args:
        - model_type: A string representing a model type (e.g., "fixed_location").
        
        Returns:
        - FarmDeliveryModelType: Corresponding enum value.
        
        Raises:
        - ValueError: If the string does not match any enum value.
        """
        try:
            return cls(model_type.lower())
        except ValueError:
            raise ValueError(f"Invalid FarmDeliveryModelType: {model_type}")

class FarmDeliveryModelMode:
    """
    Class to handle different modes of the farming delivery model.
    
    This class sets the model type and provides methods for getting a file prefix and 
    creating an environment instance based on the model type.
    """
    
    def __init__(self, model_type: FarmDeliveryModelType):
        """
        Initializes the FarmDeliveryModelMode with a given model type.
        
        Args:
        - model_type: The type of farm delivery model (e.g., fixed orders, fixed locations).
        
        Raises:
        - TypeError: If the model_type is not an instance of FarmDeliveryModelType.
        """
        if not isinstance(model_type, FarmDeliveryModelType):
            raise TypeError("model_type must be an instance of FarmDeliveryModelType")
        
        self.model_type = model_type

    def get_file_prefix(self) -> str:
        """
        Returns a file prefix based on the model type.
        
        This prefix can be used for saving files related to the model (e.g., model checkpoints).
        
        Returns:
        - str: A string file prefix corresponding to the model type.
        """
        return f"delivery_{self.model_type.value}"

    def get_environment(self) -> FarmEnv:
        """
        Returns a configured environment based on the model type.
        
        The environment is customized based on whether the delivery model is fixed or non-fixed.
        
        Returns:
        - FarmEnv: An instance of the environment configured for the specified model type.
        """
        match self.model_type:
            case FarmDeliveryModelType.FIXED_LOCATION:
                # Environment with fixed delivery location
                return FarmEnv(fix_location=True, render_mode="non_human")
            case FarmDeliveryModelType.FULLY_FIXED_STATE:
                # Environment with both fixed orders and fixed delivery locations
                return FarmEnv(fix_orders=True, fix_location=True, render_mode="non_human")
            case FarmDeliveryModelType.FIXED_ORDERS:
                # Environment with fixed orders (location is variable)
                return FarmEnv(fix_orders=True, render_mode="non_human")
            case FarmDeliveryModelType.NON_FIXED_STATE:
                # Environment with both orders and locations randomized
                return FarmEnv(render_mode="non_human")
            case _:
                # Default environment (non-fixed state)
                return FarmEnv(render_mode="non_human")
