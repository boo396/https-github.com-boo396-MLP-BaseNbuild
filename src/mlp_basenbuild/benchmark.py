from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def worker(client: httpx.AsyncClient, url: str, text: str, latencies: list[float]) -> None:
    start = time.perf_counter()
    response = await client.post(url, json={"text": text, "has_image": False})
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    latencies.append(elapsed_ms)


async def run_benchmark(url: str, requests: int, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def run_one(i: int) -> None:
            async with semaphore:
                await worker(client, url, f"benchmark prompt {i}", latencies)

        start = time.perf_counter()
        await asyncio.gather(*(run_one(i) for i in range(requests)))
        total_s = time.perf_counter() - start

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(0.50 * len(lat_sorted))]
    p95 = lat_sorted[int(0.95 * len(lat_sorted)) - 1]
    rps = requests / total_s

    print(f"requests={requests} concurrency={concurrency}")
    print(f"p50_ms={p50:.2f} p95_ms={p95:.2f} mean_ms={statistics.mean(latencies):.2f}")
    print(f"throughput_rps={rps:.2f} total_s={total_s:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/route")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
