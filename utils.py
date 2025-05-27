import json, os

def remove_dirs(cfg):
    """Remove the project directory and its contents."""
    if cfg.project_dir.exists():
        import shutil
        shutil.rmtree(cfg.project_dir)
        print(f"Folder '{cfg.project_dir}' deleted.")
    else:
        print(f"Folder '{cfg.project_dir}' does not exist.")
        
def check_and_create_dirs(cfg):
    """Ensure the project directory and chunks directory exist, creating them if necessary."""
    if not cfg.project_dir.exists():
        cfg.project_dir.mkdir(parents=True)
        cfg.chunks_dir.mkdir(parents=True)
        print(f"Folder '{cfg.project_dir}' and '{cfg.chunks_dir}' has been created.")
    else:
        print(f"Folder '{cfg.project_dir}' already exists.")

def load_metadata(cfg):
    """ 
    Load metadata from a JSON file.
    Args:
        cfg (Config): Configuration object containing metadata file path.
    Returns:
        list: List of metadata entries.
    """
    if os.path.exists(cfg.metadata_file):
        with open(cfg.metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata