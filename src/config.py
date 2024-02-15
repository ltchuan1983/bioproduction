"""
singleton template

Returns:
    [type]: [description]
    """
import yaml

class Config:
    _instance = None # class-level variable

    def __new__(cls):  # refer to the class itself
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self):
        # Load config yaml file

        try:
            self.config_data = load_config_yaml('../config.yaml')
        except:
            raise FileNotFoundError("config.yaml is not found")


def load_config_yaml(config_path):
    """ Load config yaml file
    
    Args:
        config_path (str): Name of path for config yaml file
    
    Returns:
        config (dict): Key_value pairs defined in config yaml file
    """

    with open(config_path) as file:
        config = yaml.safe_load(file)
    
    return config
