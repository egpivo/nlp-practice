FROM python:3.10-buster as builder

RUN pip install --no-cache-dir --user poetry==1.6.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache\
    PATH="${PATH}:/root/.local/bin"

WORKDIR /nlp-practice

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.10-slim-buster as runtime
ENV VIRTUAL_ENV=/nlp_practice/.venv \
    PATH="/nlp-practice/.venv/bin:$PATH" \
    PYTHONPATH="${PYTHONPATH}:/nlp-practice"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY llm /llm

ENTRYPOINT ["python", "-m", "llm.main"]
