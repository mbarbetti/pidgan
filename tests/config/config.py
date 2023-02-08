import os
from argparse import ArgumentParser

import yaml

here = os.path.dirname(__file__)
parser = ArgumentParser(description="tests configuration")

parser.add_argument("--interactive", action="store_true")
parser.add_argument("--no-interactive", dest="interactive", action="store_false")
parser.set_defaults(interactive=True)

parser.add_argument("-a", "--address", default="http://hopaas.cloud.infn.it")
parser.add_argument("-p", "--port", default=80)
parser.add_argument("-t", "--token", default="user-api-token")
config_hopaas = dict()

args = parser.parse_args()

if args.interactive:
    address = input(f"Address for the Hopaas service (default: '{args.address}'): ")
    config_hopaas["address"] = address if not (address == "") else args.address

    port = input(f"Port for the Hopaas service (default: '{args.port}'): ")
    config_hopaas["port"] = port if not (port == "") else args.port

    token = input(f"API token for the Hopaas service (default: '{args.token}'): ")
    config_hopaas["token"] = token if not (token == "") else args.token
else:
    config_hopaas["address"] = args.address
    config_hopaas["port"] = args.port
    config_hopaas["token"] = args.token

with open(f"{here}/hopaas.yml", "w") as file:
    yaml.dump(config_hopaas, file)
