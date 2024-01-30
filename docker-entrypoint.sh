#!/bin/sh

set -e

# activate the virtual environment here
. /opt/pysetup/.venv/bin/activate

# run alembic migrations


# You can put other setup logic here

# Evaluating passed command:
exec "$@"
