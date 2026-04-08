FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock ./
COPY src src

RUN uv sync --locked

ENV OPENAI_API_KEY=""
ENV AGENT_MODEL="gpt-4o-mini"

ENTRYPOINT ["uv", "run", "src/server.py", "--host", "0.0.0.0"]
CMD ["--port", "9009"]
EXPOSE 9009
