#!/usr/bin/env python3
"""Plain-Python (asyncio + raw sockets) client for the live cluster.
No frontend framework -- just the wire protocol in raft/wire.py.

    python3 scripts/kv_client.py --nodes 127.0.0.1:9101,127.0.0.1:9102,127.0.0.1:9103 put x 42
    python3 scripts/kv_client.py --nodes 127.0.0.1:9101,127.0.0.1:9102,127.0.0.1:9103 get x

Tries each address in --nodes in turn, following "not_leader" redirects
(the reply's leader_hint names a node_id, not an address, so on redirect
this falls back to round-robin over the remaining --nodes rather than
resolving the hint to an address -- good enough for a demo where the
caller already knows every node's client address).
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raft.wire import decode_client_reply, encode_client_request, read_framed

CLIENT_ID = f"cli-{int(time.time() * 1000) % 100000}"


async def _try_one(addr: str, seq: int, op: str, key: str, value: str | None, timeout_s: float):
    host, port = addr.split(":")
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), timeout=timeout_s)
    try:
        writer.write(encode_client_request(CLIENT_ID, seq, op, key, value))
        await writer.drain()
        body = await asyncio.wait_for(read_framed(reader), timeout=timeout_s)
        return decode_client_reply(body)
    finally:
        writer.close()


async def request(addrs: list[str], op: str, key: str, value: str | None = None, max_attempts: int = 10) -> dict:
    seq = int(time.time() * 1000)
    for attempt, addr in zip(range(max_attempts), itertools.cycle(addrs)):
        try:
            reply = await _try_one(addr, seq, op, key, value, timeout_s=2.0)
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            continue
        if reply.get("error") == "not_leader" and reply.get("leader_hint"):
            continue  # try the next address; likely the actual leader
        return reply
    raise RuntimeError(f"could not complete {op} {key!r} against any of {addrs} after {max_attempts} attempts")


def main() -> None:
    args = sys.argv[1:]
    if "--nodes" not in args:
        print(__doc__)
        sys.exit(1)
    nodes_idx = args.index("--nodes")
    addrs = args[nodes_idx + 1].split(",")
    rest = args[:nodes_idx] + args[nodes_idx + 2 :]
    if not rest:
        print(__doc__)
        sys.exit(1)
    op = rest[0]
    key = rest[1] if len(rest) > 1 else None
    value = rest[2] if len(rest) > 2 else None

    reply = asyncio.run(request(addrs, op, key, value))
    print(reply)


if __name__ == "__main__":
    main()
