"""
Get an estimate of the safe gating range by iterative convex polygon expansion
"""

from ..utils.tandem_sweep import SweepResult, ezTandemSweep

import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

Point = Tuple[float, float]
Edge = Tuple[Point, Point]

class ConvexPolygon():
    def __init__(self, vertices: List[Point]):
        self.vertices = vertices

    def edges(self) -> List[Edge]:
        return [(self.vertices[i], self.vertices[(i+1)%len(self.vertices)]) \
                for i in range(len(self.vertices))]

    def longest_edge(self):
         return max(self.edges(), \
                    key = lambda e: np.linalg.norm(np.subtract(e[1], e[0])))

    def includes(self, p: Point):
        vx, vy = p
        sign = None
        for e in self.edges():
            e0, e1 = e
            ex, ey = e1[0] - e0[0], e1[1] - e0[1]
            ux, uy = vx - e0[0], vy - e0[1]
            cross = ux*ey - uy*ex

            if sign is None:
                sign = np.sign(cross)
                continue

            if np.sign(cross) != sign:
                return False
        
        return True
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vertices, f, indent = 2)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.vertices = json.load(f)

@dataclass
class GateTest():
    strict_panic    : Callable[..., bool]
    lenient_panic   : Callable[..., bool]
    x_bounds        : Tuple[float, float]
    y_bounds        : Tuple[float, float]
    parms           : List[dict]
    origin          : List[float] | dict
    e0              : Point = (1.0, 0.0)
    ramp_wait       : float = 0.1
    max_vertices    : int = 20
    small_edge_size : float = None
    callback        : Callable[..., Any] = None
    logger          : object = None

    def __post_init__(self):
        self.gating_region = None
        self.safe = False

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def in_boundary_box(self, x, y) -> bool:
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax
    
    def go_to_origin(self) -> SweepResult:
        return ezTandemSweep(parms = self.parms, 
                             target = self.origin, 
                             wait = self.ramp_wait)

    def run(self):
        
        def strict_panic(v, *args, **kwargs) -> bool:
            x, y = v
            return self.in_boundary_box(x, y) and \
                self.strict_panic(v, *args, **kwargs)

        # Sweep to the origin
        self.info("Sweeping to the origin...")
        
        res = self.go_to_origin()
        if res != SweepResult.SUCCEEDED:
            raise RuntimeError("Sweep to origin failed.")
        self.info("Successfully reached the origin")
         
        get = [P['f'] for P in self.parms]
        if isinstance(self.origin, dict):
            zero = [v for k, v in self.origin.items()]
        else:
            zero = np.array(self.origin)
        e0 = np.array(self.e0)
        e0 /= np.linalg.norm(e0)

        # sweep as far as we can in the +e0 direction
        self.info("Sweeping to the positive e0 direction...")
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        long_step = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
        res = ezTandemSweep(parms = self.parms,
                            target = zero + long_step * e0,
                            wait = self.ramp_wait,
                            panic_condition = strict_panic,
                            panic_behavior = 'stop')
        
        if res == SweepResult.FAILED:
            raise RuntimeError("Sweep to positive e0 failed.")
        vertex1 = (get[0](), get[1]())
        self.info(f"Sweep to positve e0 ended at vertex: {vertex1}")

        self.info("Returning to the origin...")
        self.go_to_origin()
        
        # sweep as far as we can in the -e0 direction
        self.info("Sweeping to the negative e0 direction...")
        res = ezTandemSweep(parms = self.parms,
                            target = zero - long_step * e0,
                            wait = self.ramp_wait,
                            panic_condition = strict_panic,
                            panic_behavior = 'stop')
        
        if res == SweepResult.FAILED:
            raise RuntimeError("Sweep to negative e0 failed.")
        vertex0 = (get[0](), get[1]())
        self.info(f"Sweep to negative e0 ended at vertex: {vertex0}")
        
        self.info("Iteratively profiling convex gating region...")
        self.gating_region = ConvexPolygon([vertex0, vertex1])
        for i in range(self.max_vertices - 2):
            self.info(f"Iteration {i}:")

            # find the longest edge
            (ax, ay), (bx, by) = self.gating_region.longest_edge()
            self.info(
                f"Expanding normal to ({ax}, {ay}) --> ({bx}, {by}) edge..."
            )

            m = np.array([ax + bx, ay + by]) / 2
            t = np.array([-(by - ay), (bx - ax)])
            edgelen = np.linalg.norm(t)
            t /= edgelen # normal direction

            # check if this is a small edge so we can terminate
            if self.small_edge_size is not None and \
                edgelen <= self.small_edge_size:
                self.info("Terminated by reaching small edge size condition")
                break

            # sweep to the midpoint
            res = ezTandemSweep(parms = self.parms, 
                                target = m, 
                                wait = self.ramp_wait)
            
            if res != SweepResult.SUCCEEDED:
                raise RuntimeError("Sweep to midpoint failed.")
            
            # sweep as far as we can outward from the edge
            res = ezTandemSweep(parms = self.parms,
                                target = m + long_step * t,
                                wait = self.ramp_wait,
                                panic_condition = strict_panic,
                                panic_behavior = 'stop')
        
            if res == SweepResult.FAILED:
                raise RuntimeError("Expansion sweep failed.")

            # insert the new vertex
            vertex = (get[0](), get[1]())
            insertion_index = self.gating_region.vertices.index((ax, ay)) + 1
            self.gating_region.vertices.insert(insertion_index, vertex)
            self.info(f"Inserting new vertex at {vertex}")
            self.info(self.gating_region.vertices)

            # callback function
            if self.callback:
                self.callback(self)

        else:
            self.info("Terminated by reaching max iterations.")

        # sweep around the boundary to check for safety.
        self.safety_check()

    def safety_check(self):

        def lenient_panic(v, *args, **kwargs) -> bool:
            x, y = v
            return self.in_boundary_box(x, y) and \
                self.lenient_panic(v, *args, **kwargs)

        self.info("Performing boundary safety check...")

        for vertex in self.gating_region.vertices:
            self.info(f"Sweeping to vertex: {vertex}")
            res = ezTandemSweep(parms = self.parms,
                                target = np.array(vertex),
                                wait = self.ramp_wait,
                                panic_condition = lenient_panic,
                                panic_behavior = self.go_to_origin)
            
            if res != SweepResult.SUCCEEDED:
                self.info(f"Safety check failed.")
                self.safe = False
                return

        self.info(f"Safety check succeeded.")
        self.info("Returning to the origin...")
        self.go_to_origin()
        self.safe = True
