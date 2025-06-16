PACKAGE_NAME = 'mlx_textgen'

def env_name() -> str:
    """Get the current python environment name.

    Returns:
        str: Current python environment name.
    """
    import os
    import sys
    base = os.path.basename(sys.prefix)
    if base.lower() == 'anaconda3':
        return 'base'
    elif 'python3' in base.lower():
        return 'base'
    else:
        return base
    
def get_config_file_dir() -> str:
    """Get the directory of the package configuration file.

    Returns:
        str: Directory of the package configuration file.
    """
    import os
    return os.path.join(os.path.expanduser('~'), '.config', PACKAGE_NAME, env_name(), 'config.json')

def get_package_cache_dir() -> str:
    """Get the directory where mlx converted models and prompt cache files are stored.

    Returns:
        str: Directory where mlx converted models and prompt cache files are stored.
    """
    import os
    import json
    default_config = dict(
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', PACKAGE_NAME)
    )
    config_dir = get_config_file_dir()
    if not os.path.exists(config_dir):
        os.makedirs(os.path.dirname(config_dir), exist_ok=True)
        with open(config_dir, 'w') as f:
            json.dump(default_config, f, indent=4)
        config = default_config
    else:
        with open(config_dir, 'r') as f:
            config = json.load(f)
    os.makedirs(config['cache_dir'], exist_ok=True)
    return config['cache_dir']

def get_prompt_cache_dir() -> str:
    """Get the directory of prompt cache files.

    Returns:
        str: Directory of prompt cache files.
    """
    import os
    prompt_cache_dir = os.path.join(get_package_cache_dir(), 'prompt_cache')
    os.makedirs(prompt_cache_dir, exist_ok=True)
    return prompt_cache_dir

def set_cache_dir(cache_dir: str) -> None:
    """Set the directory where mlx converted models and prompt cache files are stored.

    Args:
        cache_dir (str): The new directory for mlx converted models and prompt cache files.
    """
    if cache_dir.strip():
        import os
        import json
        get_package_cache_dir()
        config = dict(cache_dir=os.path.abspath(cache_dir))
        with open(get_config_file_dir(), 'w') as f:
            json.dump(config, f, indent=4)
        
    else:
        import warnings
        warnings.warn("`cache_dir` cannot be None or an empty string. `cache_dir` not set.")



    

