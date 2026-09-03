FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system agentmesh && useradd --system --gid agentmesh --home /app agentmesh

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install ".[postgres]"

COPY configs ./configs
RUN chown -R agentmesh:agentmesh /app

USER agentmesh
EXPOSE 8080

CMD ["uvicorn", "agent_mesh_risk_lab.api:app", "--host", "0.0.0.0", "--port", "8080"]
