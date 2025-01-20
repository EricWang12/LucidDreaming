import yaml
import sys

def update_yaml(src_yaml_path, target_yaml_path):
    """
    Update the 'inherit_from' field in the target YAML file with the value from the source YAML file.

    Parameters:
        src_yaml_path (str): Path to the source YAML file.
        target_yaml_path (str): Path to the target YAML file.
    """
    try:
        # Load the source YAML file
        with open(src_yaml_path, 'r') as file:
            src_cfg = yaml.safe_load(file)
        
        # Get the 'inherit_from' value from the source YAML file
        parent = src_cfg.get('inherit_from', None)
        if parent is None:
            raise ValueError(f"No 'inherit_from' field found in {src_yaml_path}")
        
        # Load the target YAML file
        with open(target_yaml_path, 'r') as file:
            target_cfg = yaml.safe_load(file)
        
        # Update the 'inherit_from' value in the target YAML file
        target_cfg['inherit_from'] = parent
        
        # Write the updated YAML data back to the target file
        with open(target_yaml_path, 'w') as file:
            yaml.safe_dump(target_cfg, file)
        
        print(f"Updated 'inherit_from' in '{target_yaml_path}' with value '{parent}' from '{src_yaml_path}'.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage:
    # python update_yaml.py source_yaml_file target_yaml_file
    if len(sys.argv) != 2:
        print("Usage: python update_yaml.py <src_yaml_path>")
        sys.exit(1)
    
    src_yaml_path = sys.argv[1]
    target_yaml_path =  "configs/control/control-inheritance.yaml"
    
    update_yaml(src_yaml_path, target_yaml_path)