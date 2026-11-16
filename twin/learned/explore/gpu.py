"""GPU helpers for the 2D exploration env: Warp path tracing and CUDA-graph relaxations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import warp as wp

wp.config.quiet = True
wp.init()

SQ2 = math.sqrt(2.0)
NB = [(-1, -1, SQ2), (-1, 0, 1.0), (-1, 1, SQ2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQ2), (1, 0, 1.0), (1, 1, SQ2)]


@wp.kernel
def _path_kernel(
    D: wp.array3d(dtype=wp.float32),
    goal_r: wp.array(dtype=wp.int32),
    goal_c: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    out_r: wp.array2d(dtype=wp.int32),
    out_c: wp.array2d(dtype=wp.int32),
    length: wp.array(dtype=wp.int32),
    maxlen: int,
):
    n = wp.tid()
    H = D.shape[1]
    W = D.shape[2]
    r = goal_r[n]
    c = goal_c[n]
    out_r[n, 0] = r
    out_c[n, 0] = c
    k = int(1)
    if active[n] != 0:
        while k < maxlen:
            cur = D[n, r, c]
            if not (cur > 0.0):
                break
            best = cur
            br = r
            bc = c
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr = r + dr
                    cc = c + dc
                    if rr >= 0 and rr < H and cc >= 0 and cc < W:
                        v = D[n, rr, cc]
                        if v < best:
                            best = v
                            br = rr
                            bc = cc
            if br == r and bc == c:
                break
            r = br
            c = bc
            out_r[n, k] = r
            out_c[n, k] = c
            k += 1
    length[n] = k


def trace_paths(D: torch.Tensor, goal_r: torch.Tensor, goal_c: torch.Tensor, active: torch.Tensor, maxlen: int):
    """Descend D from each goal to the source. Returns R, C [N, L] ordered source -> goal (padded
    with the goal cell after the end) and plen [N] (number of cells incl. both ends)."""
    N = D.shape[0]
    dev = D.device
    out_r = torch.zeros(N, maxlen, dtype=torch.int32, device=dev)
    out_c = torch.zeros(N, maxlen, dtype=torch.int32, device=dev)
    length = torch.zeros(N, dtype=torch.int32, device=dev)
    wp.launch(
        _path_kernel,
        dim=N,
        inputs=[
            wp.from_torch(D.contiguous()),
            wp.from_torch(goal_r.int().contiguous()),
            wp.from_torch(goal_c.int().contiguous()),
            wp.from_torch(active.int().contiguous()),
            wp.from_torch(out_r),
            wp.from_torch(out_c),
            wp.from_torch(length),
            int(maxlen),
        ],
        device=wp.device_from_torch(dev),
    )
    plen = length.long()
    L = int(plen.max().item())
    i = torch.arange(L, device=dev)
    src = (plen[:, None] - 1 - i[None]).clamp(min=0)
    R = out_r.long().gather(1, src)
    C = out_c.long().gather(1, src)
    return R, C, plen


class GraphLoop:
    """Runs `body(state)` (in place, fixed shapes) `iters` times per CUDA-graph replay until the
    state stops changing."""

    def __init__(self, body, state: torch.Tensor, iters: int = 16):
        self.body = body
        self.state = state
        self.prev = torch.empty_like(state)
        self.iters = iters
        self.graph = None
        if state.is_cuda:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body(state)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                for _ in range(iters):
                    body(state)

    def run(self, max_replays: int):
        for _ in range(max_replays):
            self.prev.copy_(self.state)
            if self.graph is not None:
                self.graph.replay()
            else:
                for _ in range(self.iters):
                    self.body(self.state)
            if torch.equal(self.prev, self.state):
                break
        return self.state


class Relaxer:
    """Geodesic cost-to-go on [N, H, W] grids by repeated 8-neighbour relaxation (CUDA graph)."""

    def __init__(self, N, H, W, device, iters=16):
        self.N, self.H, self.W = N, H, W
        self.D = torch.full((N, H, W), float("inf"), device=device)
        self.cost = torch.ones(N, H, W, device=device)
        self.Dp = torch.full((N, H + 2, W + 2), float("inf"), device=device)
        self.loop = GraphLoop(self._body, self.D, iters)

    def _body(self, D):
        H, W = self.H, self.W
        self.Dp[:, 1:-1, 1:-1].copy_(D)
        best = D.clone()
        for dr, dc, s in NB:
            torch.minimum(best, self.Dp[:, 1 + dr:1 + dr + H, 1 + dc:1 + dc + W] + s * self.cost, out=best)
        D.copy_(best)

    def run(self, D0, cost):
        self.D.copy_(D0)
        self.cost.copy_(cost)
        self.loop.run(max_replays=(3 * self.H) // self.loop.iters + 2)
        return self.D.clone()


class Labeler:
    """8-connected component labels by max-propagation within a mask (CUDA graph)."""

    def __init__(self, N, H, W, device, iters=16):
        self.lab = torch.zeros(N, H, W, device=device)
        self.mask = torch.zeros(N, H, W, device=device)
        self.loop = GraphLoop(self._body, self.lab, iters)

    def _body(self, lab):
        lab.copy_(F.max_pool2d(lab.unsqueeze(1), 3, 1, 1).squeeze(1) * self.mask)

    def run(self, lab0, mask):
        self.lab.copy_(lab0)
        self.mask.copy_(mask.float())
        H = lab0.shape[1]
        self.loop.run(max_replays=(2 * H) // self.loop.iters + 2)
        return self.lab.clone()
