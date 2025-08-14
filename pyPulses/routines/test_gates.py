"""
Get an estimate of the safe gating range by iterative convex polygon expansion
"""

from ..utils.tandem_sweep import SweepResult, ezTandemSweep
from ..utils.getsetter import getSetter
from ..plotting.chart_recorder import Recorder, Line, Polygon

import json
import numpy as np
from copy import deepcopy
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

    def get_scatter_points(self) -> Tuple[List[float], List[float]]:
        return [[*x, x[-1]] for x in zip(*self.vertices)]

@dataclass
class GateTest():
    strict_panic    : Callable[..., bool]
    lenient_panic   : Callable[..., bool]
    x_bounds        : Tuple[float, float]
    y_bounds        : Tuple[float, float]
    parameters      : List[dict]
    origin          : List[float] | dict
    e0              : Point = (1.0, 0.0)
    ramp_wait       : float = 0.1
    max_vertices    : int = 20
    small_edge_size : float = None
    plot            : bool = False
    danger_callback : bool = False
    callback        : Callable[..., Any] = None
    logger          : object = None

    def __post_init__(self):
        self.gating_region = None
        self.safe = False

        if isinstance(self.origin, dict):
            self.zero = [self.origin['x'], self.origin['y']]
        else:
            self.zero = np.array(self.origin)

        def bounded_strict_panic(v, *args, **kwargs) -> bool:
            x, y = v
            return (not self.in_boundary_box(x, y)) or \
                self.strict_panic(v, *args, **kwargs)
        self.bounded_strict_panic = bounded_strict_panic

        def bounded_lenient_panic(v, *args, **kwargs) -> bool:
            x, y = v
            return (not self.in_boundary_box(x, y)) or \
                self.lenient_panic(v, *args, **kwargs)
        self.bounded_lenient_panic = bounded_lenient_panic

        self.parms = deepcopy(self.parameters)
        for p in self.parms:
            if not 'f' in p:
                p['f'] = getSetter(p['get'], p['set'])
        
        # set up plotter
        if self.plot:
            ncols = 2 if self.danger_callback else 1
            self.plotter = Recorder(rows=1, cols=ncols, width=1000, height=500)
            self.plotter.add_element(Polygon('poly', color='green'), row=1, col=ncols)
            self.plotter.add_element(Line('pos', color='red'), row=1, col=ncols)
            self.plotter.set_axis_labels(row=1, col=ncols, xlabel='x', ylabel='y')
            self.plotter.set_axis_range(row=1, col=ncols, xrange=self.x_bounds,
                                        yrange=self.y_bounds)
            
            if self.danger_callback:
                self.plotter.add_element(Line('danger', color='blue'), row=1, col=1)
                self.plotter.set_axis_labels(row=1, col=1, xlabel='N', 
                                            ylabel='Danger Parameter')
                self.bounded_strict_panic = lambda v, *args, **kwargs: \
                    bounded_strict_panic(v, *args, callback = \
                                         self._plot_danger_callback, **kwargs)
                self.bounded_lenient_panic = lambda v, *args, **kwargs: \
                    bounded_lenient_panic(v, *args, callback = \
                                          self._plot_danger_callback, **kwargs)
                self.ndanger = 0
            
    def _plot_position_callback(self, pos: np.ndarray):
        if not self.plot:
            return
        self.plotter.update({'pos': {'x_new': pos[0], 'y_new': pos[1]}})

    def _plot_danger_callback(self, danger_lvl: float):
        if not self.plot:
            return
        self.ndanger += 1
        self.plotter.update({'danger': {'x_new': self.ndanger, 'y_new': danger_lvl}})

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def in_boundary_box(self, x, y) -> bool:
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        return xmin <= x <= xmax and ymin <= y <= ymax
    
    def go_to_origin(self) -> SweepResult:
        return ezTandemSweep(parms = self.parms, 
                             target = self.zero, 
                             wait = self.ramp_wait,
                             callback = self._plot_position_callback)

    def run(self):
        
        if self.plot:
            self.plotter.show()

        # Sweep to the origin
        self.info("Sweeping to the origin...")
        
        res = self.go_to_origin()
        if res != SweepResult.SUCCEEDED:
            raise RuntimeError("Sweep to origin failed.")
        self.info("Successfully reached the origin")
         
        get = [P['f'] for P in self.parms]
        e0 = np.array(self.e0)
        e0 /= np.linalg.norm(e0)

        # sweep as far as we can in the +e0 direction
        self.info("Sweeping to the positive e0 direction...")
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        long_step = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
        res = ezTandemSweep(parms = self.parms,
                            target = self.zero + long_step * e0,
                            wait = self.ramp_wait,
                            callback = self._plot_position_callback,
                            panic_condition = self.bounded_strict_panic,
                            panic_behavior = 'stop')
        
        if res == SweepResult.FAILED:
            raise RuntimeError("Sweep to positive e0 failed.")
        vertex1 = (np.clip(get[0](), xmin, xmax), np.clip(get[1](), ymin, ymax))
        self.info(f"Sweep to positve e0 ended at vertex: {vertex1}")

        self.info("Returning to the origin...")
        self.go_to_origin()
        
        # sweep as far as we can in the -e0 direction
        self.info("Sweeping to the negative e0 direction...")
        res = ezTandemSweep(parms = self.parms,
                            target = self.zero - long_step * e0,
                            wait = self.ramp_wait,
                            callback = self._plot_position_callback,
                            panic_condition = self.bounded_strict_panic,
                            panic_behavior = 'stop')
        
        if res == SweepResult.FAILED:
            raise RuntimeError("Sweep to negative e0 failed.")
        vertex0 = (np.clip(get[0](), xmin, xmax), np.clip(get[1](), ymin, ymax))
        self.info(f"Sweep to negative e0 ended at vertex: {vertex0}")
        
        self.info("Iteratively profiling convex gating region...")
        self.gating_region = ConvexPolygon([vertex0, vertex1])
        
        # iteratively expand the gating region.
        self.iteratively_expand()

        # sweep around the boundary to check for safety.
        self.safety_check()

        if self.plot:
            self.plotter.draw()

    def iteratively_expand(self):

        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        long_step = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
        get = [P['f'] for P in self.parms]

        # we want to expand from the longest edge not on the boundary box
        def grade_edge(e: Edge):
            if e[0][0] == e[1][0] and e[1][0] in self.x_bounds:
                return 0.0
            if e[0][1] == e[1][1] and e[1][1] in self.y_bounds:
                return 0.0
            return np.linalg.norm(np.subtract(e[1], e[0]))

        def get_edge():
            return max(self.gating_region.edges(), key = grade_edge)

        for i in range(len(self.gating_region.vertices), self.max_vertices):
            self.info(f"Current Number of Edges: {i + 1}:")

            # find the longest edge
            (ax, ay), (bx, by) = get_edge()
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
                                wait = self.ramp_wait,
                                callback = self._plot_position_callback)
            
            if res != SweepResult.SUCCEEDED:
                raise RuntimeError("Sweep to midpoint failed.")
            
            # sweep as far as we can outward from the edge
            res = ezTandemSweep(parms = self.parms,
                                target = m + long_step * t,
                                wait = self.ramp_wait,
                                callback = self._plot_position_callback,
                                panic_condition = self.bounded_strict_panic,
                                panic_behavior = 'stop')
        
            if res == SweepResult.FAILED:
                raise RuntimeError("Expansion sweep failed.")

            # insert the new vertex
            vertex = (np.clip(get[0](), xmin, xmax), 
                      np.clip(get[1](), ymin, ymax))
            insertion_index = self.gating_region.vertices.index((ax, ay)) + 1
            self.gating_region.vertices.insert(insertion_index, vertex)
            self.info(f"Inserting new vertex at {vertex}")

            if self.plot:
                self.plotter.elements['poly'].update(
                    self.plotter.widget, self.gating_region.vertices)
                self.plotter.draw()

            # callback function
            if self.callback:
                self.callback(self)

        else:
            self.info("Terminated by reaching max iterations.")

    def safety_check(self):

        self.info("Returning to the origin...")
        self.go_to_origin()

        self.info("Performing boundary safety check...")

        for vertex in self.gating_region.vertices:
            self.info(f"Sweeping to vertex: {vertex}")
            res = ezTandemSweep(parms = self.parms,
                                target = np.array(vertex),
                                wait = self.ramp_wait,
                                callback = self._plot_position_callback,
                                panic_condition = self.bounded_lenient_panic,
                                panic_behavior = \
                                    lambda *args, **kwargs: self.go_to_origin)
            
            if res != SweepResult.SUCCEEDED:
                self.info(f"Safety check failed.")
                self.safe = False
                return

        self.info(f"Safety check succeeded.")
        self.info("Returning to the origin...")
        self.go_to_origin()
        self.safe = True
