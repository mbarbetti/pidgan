import os

import yaml

here = os.path.abspath(os.path.dirname(__file__))

# +-------------------+
# |   Initial setup   |
# +-------------------+

with open(f"{here}/config/directories.yml") as file:
    config_dir = yaml.full_load(file)


def log_message(dirname):
    print(f"[INFO] Directory '{dirname}' successfully created")


# +--------------------------+
# |   Directories creation   |
# +--------------------------+

# Deault directories
for key in config_dir.keys():
    dir_ = config_dir[key]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        log_message(dir_)

# Directories for optimization studies
for key in ["models_dir", "images_dir", "reports_dir"]:
    opt_dir = f"{config_dir[key]}/opt_studies"
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
        log_message(opt_dir)

# Directories for logs and tests
for dir_ in [f"{here}/logs", f"{here}/exports/logs", f"{here}/exports/images"]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        log_message(dir_)
