"""Multiplication worker. Takes a batch of (tile, k-slice) jobs, returns partials."""

import argparse
import socketserver
import numpy as np

from shared import recv_msg, send_msg


class WorkerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        msg = recv_msg(self.request)
        results = []
        for job in msg["assignments"]:
            a_block = np.array(job["A_block"], dtype=float)
            b_block = np.array(job["B_block"], dtype=float)
            k_start, k_end = job["k_start"], job["k_end"]

            # only this worker's assigned k-slice, not necessarily the whole tile
            partial = a_block[:, k_start:k_end] @ b_block[k_start:k_end, :]

            results.append({
                "tile_row": job["tile_row"],
                "tile_col": job["tile_col"],
                "k_start": k_start,
                "k_end": k_end,
                "partial": partial.tolist(),
            })
        send_msg(self.request, {"results": results})


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Stream-K multiplication worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = ThreadingTCPServer((args.host, args.port), WorkerHandler)
    print(f"[worker] listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
