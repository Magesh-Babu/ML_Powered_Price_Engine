import os
import pickle
from config import config

# Only import cryptography if encryption is enabled, to avoid unnecessary dependency if not used
fernet = None
if config['encryption'].get('enabled'):
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise ImportError("Encryption is enabled but the 'cryptography' library is not installed") from e
    key_str = config['encryption'].get('key')
    if not key_str:
        raise RuntimeError("Encryption is enabled but no key provided in config")
    # Prepare Fernet cipher for encryption/decryption
    fernet = Fernet(key_str.encode())

def ensure_user_dirs(user_id: str):
    """Ensure that the necessary directories for a user's data and models exist."""
    base_dir = config['storage']['base_dir']
    os.makedirs(f"{base_dir}/{user_id}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/{user_id}/logs", exist_ok=True)

def get_latest_model_path(user_id: str):
    """
    Retrieves the file path and version number of the latest saved model for a user.

    Args:
        user_id (str): Identifier for the user.

    Returns:
        tuple: (str or None, int or None) - Path to the latest model file and its version, 
               or (None, None) if no model exists.
    """
    base_dir = config['storage']['base_dir']
    models_dir = f"{base_dir}/{user_id}/models"
    if not os.path.isdir(models_dir):
        return None, None
    # Find all model files of pattern model_v*.pkl
    files = [f for f in os.listdir(models_dir) if f.startswith("model_v") and f.endswith(".pkl")]
    if not files:
        return None, None
    # Extract version numbers and find the max
    versions = []
    for fname in files:
        try:
            ver = int(fname.split("model_v")[1].split(".pkl")[0])
            versions.append(ver)
        except ValueError:
            continue
    if not versions:
        return None, None
    latest_ver = max(versions)
    latest_path = os.path.join(models_dir, f"model_v{latest_ver}.pkl")
    return latest_path, latest_ver

def save_model(user_id: str, obj, component: str = "model") -> int:
    """
    Saves a versioned component (e.g., model or encoder) to the user's model directory.

    Args:
        user_id (str): Identifier for the user.
        obj: The object to be serialized and saved (e.g., model, encoder, metadata).
        component (str): Component name (e.g., 'model', 'freq_encoder').

    Returns:
        int: The new version number assigned to the saved component.
    """
    ensure_user_dirs(user_id)
    base_dir = config['storage']['base_dir']
    models_dir = os.path.join(base_dir, user_id, "models")

    # Find latest version of this component
    existing_versions = []
    for fname in os.listdir(models_dir):
        if fname.startswith(f"{component}_v") and fname.endswith(".pkl"):
            try:
                ver = int(fname.split("_v")[1].split(".pkl")[0])
                existing_versions.append(ver)
            except:
                continue
    new_ver = max(existing_versions) + 1 if existing_versions else 1

    file_path = os.path.join(models_dir, f"{component}_v{new_ver}.pkl")

    # Serialize
    data = pickle.dumps(obj)

    # Encrypt if configured
    if config['encryption'].get('enabled') and fernet:
        data = fernet.encrypt(data)

    # Save to disk
    with open(file_path, 'wb') as f:
        f.write(data)

    return new_ver


def load_model(user_id: str, component: str = "model", version: int = None):
    """
    Loads a versioned model or component (e.g., encoder) for a specific user.

    Args:
        user_id (str): Identifier for the user.
        component (str): Component name to load (e.g., 'model', 'freq_encoder').
        version (int, optional): Specific version to load. If None, the latest is loaded.

    Returns:
        tuple: (Loaded object, int) - The deserialized component and its version number.

    Raises:
        FileNotFoundError: If the specified component or version does not exist.
    """
    base_dir = config['storage']['base_dir']
    component_dir = os.path.join(base_dir, user_id, "models")

    if version is None:
        # Auto-detect latest version
        versions = []
        for fname in os.listdir(component_dir):
            if fname.startswith(f"{component}_v") and fname.endswith(".pkl"):
                try:
                    ver = int(fname.split("_v")[1].split(".pkl")[0])
                    versions.append((ver, fname))
                except:
                    continue
        if not versions:
            raise FileNotFoundError(f"No {component} found for user '{user_id}'")
        latest_ver, latest_fname = max(versions, key=lambda x: x[0])
        model_path = os.path.join(component_dir, latest_fname)
        version = latest_ver
    else:
        model_path = os.path.join(component_dir, f"{component}_v{version}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{component} version {version} not found for user '{user_id}'")

    with open(model_path, 'rb') as f:
        data = f.read()

    if config['encryption'].get('enabled') and fernet:
        data = fernet.decrypt(data)

    return pickle.loads(data), version
