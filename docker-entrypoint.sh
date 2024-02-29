#!/bin/sh

set -e

# activate the virtual environment here
. /opt/pysetup/.venv/bin/activate

# You can put other setup logic here

# Evaluating passed command:
exec "$@"
