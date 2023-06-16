import os
from argparse import ArgumentParser

import yaml

here = os.path.dirname(__file__)
parser = ArgumentParser(description="scripts configuration")

parser.add_argument("--interactive", action="store_true")
parser.add_argument("--no-interactive", dest="interactive", action="store_false")
parser.set_defaults(interactive=True)

parser.add_argument("-D", "--data_dir", default="./data")
parser.add_argument("-M", "--models_dir", default="./models")
parser.add_argument("-I", "--images_dir", default="./images")
parser.add_argument("-R", "--reports_dir", default="./html")
config_dir = dict()

parser.add_argument(
    "-s", "--server", default="http://hopaas.cloud.infn.it:11756"
)  # https://hopaas.cloud.infn.it
parser.add_argument("-t", "--token", default="user-api-token")
config_hopaas = dict()

args = parser.parse_args()

if args.interactive:
    data_dir = input(f"Path for the data directory (default: '{args.data_dir}'): ")
    config_dir["data_dir"] = data_dir if not (data_dir == "") else args.data_dir

    models_dir = input(
        f"Path for the models directory (default: '{args.models_dir}'): "
    )
    config_dir["models_dir"] = models_dir if not (models_dir == "") else args.models_dir

    images_dir = input(
        f"Path for the images directory (default: '{args.images_dir}'): "
    )
    config_dir["images_dir"] = images_dir if not (images_dir == "") else args.images_dir

    reports_dir = input(
        f"Path for the reports directory (default: '{args.reports_dir}'): "
    )
    config_dir["reports_dir"] = (
        reports_dir if not (reports_dir == "") else args.reports_dir
    )

    server = input(f"Address of the Hopaas service (default: '{args.server}'): ")
    config_hopaas["server"] = server if not (server == "") else args.server

    token = input(f"API token to access the Hopaas service (default: '{args.token}'): ")
    config_hopaas["token"] = token if not (token == "") else args.token
else:
    config_dir["data_dir"] = args.data_dir
    config_dir["models_dir"] = args.models_dir
    config_dir["images_dir"] = args.images_dir
    config_dir["reports_dir"] = args.reports_dir

    config_hopaas["server"] = args.server
    config_hopaas["token"] = args.token

with open(f"{here}/directories.yml", "w") as file:
    yaml.dump(config_dir, file)

with open(f"{here}/hopaas.yml", "w") as file:
    yaml.dump(config_hopaas, file)
