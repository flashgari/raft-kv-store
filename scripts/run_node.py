#!/usr/bin/env python3
"""Run one Raft cluster node as a real OS process with real TCP sockets.

    python3 scripts/run_node.py --id n1 --port 9001 --client-port 9101 \\
        --peers n1=127.0.0.1:9001,n2=127.0.0.1:9002,n3=127.0.0.1:9003 \\
        --data-dir data/live_demo

Run one of these per node (see scripts/run_demo.py for a 3-node
orchestrated demo that starts, exercises, and kills nodes automatically).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raft.transport import AsyncRaftServer


def parse_peers(spec: str) -> dict[str, tuple[str, int]]:
    peers = {}
    for part in spec.split(","):
        node_id, addr = part.split("=")
        host, port = addr.split(":")
        peers[node_id] = (host, int(port))
    return peers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--port", type=int, required=True, help="peer-to-peer listen port")
    parser.add_argument("--client-port", type=int, required=True)
    parser.add_argument("--peers", required=True, help="id=host:port,id=host:port,... (including self)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/live_demo"))
    args = parser.parse_args()

    all_peers = parse_peers(args.peers)
    peer_addrs = {nid: addr for nid, addr in all_peers.items() if nid != args.id}

    server = AsyncRaftServer(
        node_id=args.id,
        peer_addrs=peer_addrs,
        listen_port=args.port,
        client_port=args.client_port,
        data_dir=args.data_dir,
    )
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
