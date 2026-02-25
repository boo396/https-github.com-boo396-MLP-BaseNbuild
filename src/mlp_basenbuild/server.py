from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, load_config
from .dispatch import WorkerDispatcher
from .models import InferenceBundle, build_inference_bundle
from .router import RouteRequest, RouteResponse, route_request


_CPU_PREV_TOTAL: float | None = None
_CPU_PREV_IDLE: float | None = None


def _normalize_percent(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return max(0.0, min(100.0, value * 100.0))
    return max(0.0, min(100.0, value))


def _collect_local_memory_stats() -> tuple[float | None, float | None, float | None]:
    try:
        mem_total_kb: int | None = None
        mem_available_kb: int | None = None
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])

        if not mem_total_kb or mem_available_kb is None:
            return None, None, None

        used_ratio = (mem_total_kb - mem_available_kb) / mem_total_kb
        mem_total_gb = mem_total_kb / (1024 * 1024)
        mem_used_gb = (mem_total_kb - mem_available_kb) / (1024 * 1024)
        return _normalize_percent(used_ratio * 100.0), mem_used_gb, mem_total_gb
    except Exception:
        return None, None, None


def _collect_local_gpu_percent() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not values:
            return None
        gpu_values = [float(value) for value in values]
        return _normalize_percent(max(gpu_values))
    except Exception:
        return None


def _collect_local_cpu_percent() -> float | None:
    global _CPU_PREV_TOTAL, _CPU_PREV_IDLE

    try:
        with open("/proc/stat", "r", encoding="utf-8") as proc_stat:
            first_line = proc_stat.readline().strip()

        parts = first_line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None

        values = [float(value) for value in parts[1:9]]
        user, nice, system, idle, iowait, irq, softirq, steal = values

        idle_all = idle + iowait
        non_idle = user + nice + system + irq + softirq + steal
        total = idle_all + non_idle

        if _CPU_PREV_TOTAL is None or _CPU_PREV_IDLE is None:
            _CPU_PREV_TOTAL = total
            _CPU_PREV_IDLE = idle_all
            return None

        total_delta = total - _CPU_PREV_TOTAL
        idle_delta = idle_all - _CPU_PREV_IDLE

        _CPU_PREV_TOTAL = total
        _CPU_PREV_IDLE = idle_all

        if total_delta <= 0:
            return None

        cpu_ratio = (total_delta - idle_delta) / total_delta
        return _normalize_percent(cpu_ratio * 100.0)
    except Exception:
        return None


def _collect_local_cpu_clock_stats() -> tuple[float | None, float | None]:
    try:
        current_values: list[float] = []
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("cpu mhz"):
                    current_values.append(float(line.split(":", 1)[1].strip()))

        current_mhz = sum(current_values) / len(current_values) if current_values else None

        if current_mhz is None:
            scaling_cur_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            cpuinfo_cur_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq")
            for cur_path in (scaling_cur_path, cpuinfo_cur_path):
                if cur_path.exists():
                    try:
                        current_khz = float(cur_path.read_text(encoding="utf-8").strip())
                        current_mhz = current_khz / 1000.0
                        break
                    except Exception:
                        continue

        max_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        max_mhz: float | None = None
        if max_freq_path.exists():
            max_khz = float(max_freq_path.read_text(encoding="utf-8").strip())
            max_mhz = max_khz / 1000.0

        if max_mhz is None:
            scaling_max_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            if scaling_max_path.exists():
                try:
                    max_khz = float(scaling_max_path.read_text(encoding="utf-8").strip())
                    max_mhz = max_khz / 1000.0
                except Exception:
                    pass

        if max_mhz is None and current_mhz is not None:
            max_mhz = current_mhz

        return current_mhz, max_mhz
    except Exception:
        return None, None


def _collect_local_gpu_clock_stats() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,clocks.max.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None, None

        first = lines[0].split(",")
        if len(first) < 2:
            return None, None

        current_mhz = float(first[0].strip())
        max_mhz = float(first[1].strip())
        return current_mhz, max_mhz
    except Exception:
        return None, None


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="MLP BaseNbuild", version="0.1.0")
    bundle: InferenceBundle = build_inference_bundle(config.mlp, len(config.routing.model_names))
    dispatcher = WorkerDispatcher(config)
    static_dir = Path(__file__).resolve().parents[2] / "static"

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "ts": int(time.time()), "backend": bundle.backend, "device": str(bundle.device)}

    @app.post("/route", response_model=RouteResponse)
    async def route(req: RouteRequest) -> RouteResponse:
        route_resp = route_request(req, config, bundle)
        if not config.routing.enable_worker_dispatch:
            route_resp.worker_status = "dispatch_disabled"
            return route_resp
        return await dispatcher.dispatch(req, route_resp)

    @app.get("/telemetry/snapshot")
    async def telemetry_snapshot() -> dict[str, object]:
        local_memory_percent, local_memory_used_gb, local_memory_total_gb = _collect_local_memory_stats()
        local_gpu_percent = _collect_local_gpu_percent()
        local_cpu_percent = _collect_local_cpu_percent()
        cpu_clock_mhz, cpu_clock_max_mhz = _collect_local_cpu_clock_stats()
        gpu_clock_mhz, gpu_clock_max_mhz = _collect_local_gpu_clock_stats()

        if (
            local_memory_percent is not None
            or local_gpu_percent is not None
            or local_cpu_percent is not None
            or cpu_clock_mhz is not None
            or gpu_clock_mhz is not None
        ):
            return {
                "ok": True,
                "source": "local_system",
                "memory_percent": local_memory_percent,
                "memory_used_gb": local_memory_used_gb,
                "memory_total_gb": local_memory_total_gb,
                "gpu_percent": local_gpu_percent,
                "cpu_percent": local_cpu_percent,
                "cpu_clock_mhz": cpu_clock_mhz,
                "cpu_clock_max_mhz": cpu_clock_max_mhz,
                "gpu_clock_mhz": gpu_clock_mhz,
                "gpu_clock_max_mhz": gpu_clock_max_mhz,
                "auth_mode": "local_only",
                "ts": int(time.time()),
            }

        return {
            "ok": False,
            "source": None,
            "memory_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
            "gpu_percent": None,
            "cpu_percent": None,
            "cpu_clock_mhz": None,
            "cpu_clock_max_mhz": None,
            "gpu_clock_mhz": None,
            "gpu_clock_max_mhz": None,
            "auth_mode": "none",
            "ts": int(time.time()),
        }

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.arm.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
