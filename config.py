import os
import yaml

# Load configuration from YAML
#config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.yaml')
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
print(f"Loading configuration from: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Validate essential config fields exist
required_sections = ['features', 'encryption', 'storage']
for section in required_sections:
    if section not in config:
        raise KeyError(f"Missing '{section}' section in config.yaml")
