#!/usr/bin/python3
import asyncio
import argparse
import json
import csv

async def monitor(cmd, interval):
    t = 0
    while True:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        yield t, json.loads(stdout.decode())
        await asyncio.sleep(interval)
        t += interval

async def run_and_monitor_temps(proc, interval):
    results = []
    async def collect_temp_data():
        async for data in monitor("sensors -j -A", interval):
            results.append(data)
    task = asyncio.create_task(collect_temp_data())
    stdout, stderr = await proc.communicate()
    if stdout: print(stdout)
    return results

async def run_and_monitor(proc, interval):
    temp_results = []
    usage_results = []

    async def collect_temp_data():
        async for data in monitor("sensors -j -A", interval):
            temp_results.append(data)
    task = asyncio.create_task(collect_temp_data())

    async def collect_usage_data():
        async for data in monitor("rocm-smi -u --showmemuse --json", interval):
            usage_results.append(data)
    task = asyncio.create_task(collect_usage_data())

    stdout, stderr = await proc.communicate()
    if stdout: print(stdout)
    return temp_results, usage_results
    

async def main(args):
    proc = await asyncio.create_subprocess_shell(" ".join(args.command))
    temp_results, usage_results = await run_and_monitor(proc, args.interval)
    with open("monitor.csv", "w", newline='') as file:
        writer = csv.writer(file, dialect="excel")
        writer.writerow(["time", "edge", "junction", "mem", "gpu_usage", "gpu_memory_usage"])
        for temp_data, gpu_data in zip(temp_results, usage_results):
            temp = temp_data[1]["amdgpu-pci-4400"]
            edge_temp = temp["edge"]["temp1_input"]
            junction_temp = temp["junction"]["temp2_input"]
            mem_temp = temp["mem"]["temp3_input"]

            gpu = gpu_data[1]["card0"]
            usage = gpu["GPU use (%)"]
            memory_usage = gpu["GPU memory use (%)"]

            writer.writerow([temp_data[0], edge_temp, junction_temp, mem_temp, usage, memory_usage])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", "-c", nargs="+")
    parser.add_argument("--interval", "-i", type=int)
    args = parser.parse_args()
    asyncio.run(main(args))




    
