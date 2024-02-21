# Creating a python base with shared environment variables

FROM python:3.10-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip


# builder-base is used to build dependencies
FROM python-base as builder-base
RUN buildDeps="build-essential" \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        vim \
    && apt-get install -y --no-install-recommends $buildDeps \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry - respects $POETRY_HOME
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | python && \
    chmod a+x /opt/poetry/bin/poetry

# We copy our Python requirements here to cache them
# and install only runtime deps using poetry
WORKDIR $PYSETUP_PATH
COPY pyproject.toml ./
RUN poetry install --only main  # respects


FROM python-base as environment
ENV SKAPP_ENV=environment

COPY --from=builder-base "$VENV_PATH" "$VENV_PATH"

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create user
RUN groupadd -g 1500 skapp && \
    useradd -m -u 1500 -g skapp skapp && \
    mkdir -p src

RUN chown -R skapp:skapp src

USER skapp
WORKDIR /src
COPY app ./app

ENTRYPOINT /docker-entrypoint.sh $0 $@
EXPOSE 8080 8000
CMD ["uvicorn", "--reload", "--proxy-headers", "--host=0.0.0.0", "--port=8080", "app.main:app"]