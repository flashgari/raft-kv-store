#!/usr/bin/env python3
"""End-to-end live demo: spawns 3 real node PROCESSES (real TCP sockets,
real OS scheduling -- not the simulated cluster the test suite uses),
writes a key, kills the leader process outright, and shows a write still
succeeds against whichever node took over.

    python3 scripts/run_demo.py

Exits non-zero and prints a diagnostic if any step fails, so this can
also be used as a real (if slow, ~15s) end-to-end smoke test of the
socket/process layer -- everything scripts/run_node.py and
scripts/kv_client.py actually touch that the simulated-cluster test suite
necessarily can't (real socket I/O, real process death, real OS
scheduling jitter).
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = ["n1", "n2", "n3"]
PEER_PORTS = {"n1": 9301, "n2": 9302, "n3": 9303}
CLIENT_PORTS = {"n1": 9401, "n2": 9402, "n3": 9403}
DATA_DIR = ROOT / "data" / "live_demo"
CLIENT_ADDRS = [f"127.0.0.1:{CLIENT_PORTS[n]}" for n in NODE_IDS]


def peers_spec() -> str:
    return ",".join(f"{n}=127.0.0.1:{PEER_PORTS[n]}" for n in NODE_IDS)


def start_node(node_id: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_node.py"),
        "--id", node_id,
        "--port", str(PEER_PORTS[node_id]),
        "--client-port", str(CLIENT_PORTS[node_id]),
        "--peers", peers_spec(),
        "--data-dir", str(DATA_DIR),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


from scripts.kv_client import request as kv_request  # noqa: E402  (needs sys.path insert above first)


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)  # fresh cluster, no state from a previous demo run

    print("Starting 3 node processes ...")
    procs = {nid: start_node(nid) for nid in NODE_IDS}
    time.sleep(2.5)  # let leader election settle

    try:
        print("PUT x=hello-raft ...")
        reply = asyncio.run(kv_request(CLIENT_ADDRS, "put", "x", "hello-raft"))
        print(" ->", reply)
        assert reply.get("ok") is True, "initial PUT failed"

        print("GET x ...")
        reply = asyncio.run(kv_request(CLIENT_ADDRS, "get", "x"))
        print(" ->", reply)
        assert reply == {"ok": True, "value": "hello-raft", "existed": True, "leader_hint": reply["leader_hint"], "error": None}

        leader_hint = reply["leader_hint"]
        print(f"Killing leader process {leader_hint} outright (SIGKILL) ...")
        procs[leader_hint].kill()
        procs[leader_hint].wait()
        del procs[leader_hint]
        time.sleep(2.0)  # let the remaining nodes elect a new leader

        remaining_addrs = [f"127.0.0.1:{CLIENT_PORTS[n]}" for n in NODE_IDS if n != leader_hint]
        print("PUT y=survived-leader-crash against the survivors ...")
        reply2 = asyncio.run(kv_request(remaining_addrs, "put", "y", "survived-leader-crash"))
        print(" ->", reply2)
        assert reply2.get("ok") is True, "PUT after leader crash failed"

        print("GET x again, to confirm the pre-crash write also survived ...")
        reply3 = asyncio.run(kv_request(remaining_addrs, "get", "x"))
        print(" ->", reply3)
        assert reply3.get("value") == "hello-raft"

        print("\nDEMO PASSED: cluster served writes, survived a killed leader, and never lost a committed key.")
    finally:
        for p in procs.values():
            p.kill()
        for p in procs.values():
            p.wait()


if __name__ == "__main__":
    main()
