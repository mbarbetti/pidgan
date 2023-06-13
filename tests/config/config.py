import os
from argparse import ArgumentParser

import yaml

here = os.path.dirname(__file__)
parser = ArgumentParser(description="tests configuration")

parser.add_argument("--interactive", action="store_true")
parser.add_argument("--no-interactive", dest="interactive", action="store_false")
parser.set_defaults(interactive=True)

parser.add_argument(
    "-s", "--server", default="http://hopaas.cloud.infn.it:11756"
)  # https://hopaas.cloud.infn.it
parser.add_argument("-t", "--token", default="user-api-token")
config_hopaas = dict()

args = parser.parse_args()

if args.interactive:
    server = input(f"Address of the Hopaas service (default: '{args.server}'): ")
    config_hopaas["server"] = server if not (server == "") else args.server

    token = input(f"API token to access the Hopaas service (default: '{args.token}'): ")
    config_hopaas["token"] = token if not (token == "") else args.token
else:
    config_hopaas["server"] = args.server
    config_hopaas["token"] = args.token

with open(f"{here}/hopaas.yml", "w") as file:
    yaml.dump(config_hopaas, file)
