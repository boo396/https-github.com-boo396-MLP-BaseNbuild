FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e .
EXPOSE 8090
CMD ["/app/.venv/bin/python", "-m", "mlp_basenbuild.server", "--config", "configs/config.arm.yaml"]
