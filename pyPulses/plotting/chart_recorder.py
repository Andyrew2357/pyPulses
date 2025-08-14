import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML
try:
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))
except:
    pass

import time
import numpy as np
from math import sqrt, ceil
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class PlotElement(ABC):
    """Base class for self-managing plot elements"""
    key: str

    @abstractmethod
    def initialize(self, widget: go.FigureWidget, row: int, col: int): ...

    @abstractmethod
    def update(self, **kwargs): ...

    @abstractmethod
    def draw(self, widget: go.FigureWidget): ...

    @abstractmethod
    def clear(self, widget: go.FigureWidget): ...

class Line(PlotElement):
    """Self-managing line element"""
    def __init__(self, key: str, name: str = None, color: str = None, 
                 width: int = 2, dash: str = 'solid', opacity: float = 1.0, 
                 secondary_y: bool = False, max_points: int = 500, 
                 legend: bool = False, **kwargs):
        self.key     = key
        self.name    = name or key
        self.color   = color
        self.width   = width
        self.dash    = dash
        self.opacity = opacity
        self.legend  = legend
        self.secondary_y = secondary_y
        self.max_points  = max_points

        self.x_data = []
        self.y_data = []
        self.trace_index = None

    def initialize(self, widget: go.FigureWidget, row: int, col: int):
        """Add this line's trace to the figure"""

        widget.add_trace(
            go.Scatter(
                x = self.x_data,
                y = self.y_data,
                name = self.name,
                mode = 'lines',
                line = {
                    'color': self.color,
                    'width': self.width,
                    'dash' : self.dash
                },
                opacity = self.opacity,
                showlegend = self.legend
            ),
            secondary_y = self.secondary_y,
            row = row, 
            col = col
        )

        # Store the trace index for future updates
        self.trace_index = len(widget.data) - 1

    def update(self, x_new: float, y_new: float):
        """Update line data"""

        if self.trace_index is None:
            raise RuntimeError(f"Line '{self.name}' not initialized.")
        self.x_data.append(x_new)
        self.y_data.append(y_new)

    def draw(self, widget: go.FigureWidget):
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]
        widget.data[self.trace_index].x = self.x_data
        widget.data[self.trace_index].y = self.y_data

    def clear(self, widget: go.FigureWidget):
        if self.trace_index is None:
            return
        
        self.x_data = []
        self.y_data = []
        widget.data[self.trace_index].x = []
        widget.data[self.trace_index].y = []

class Polygon(Line):
    def initialize(self, widget: go.FigureWidget, row: int, col: int):
        super().initialize(widget, row, col)
        widget.data[self.trace_index].fill = 'toself'

    def update(self, widget: go.FigureWidget, 
               vertices: List[Tuple[float, float]]):
        self.x_data, self.y_data = [[*x, x[-1]] for x in zip(*vertices)]

class Recorder():
    """Manages subplots and coordinates for live plotting"""
    def __init__(self, rows: int = 1, cols: int = 1, 
                 width: int = 1400, height: int = 800, 
                 draw_interval: float = 0.2):
        
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.draw_interval = draw_interval

        self.elements: Dict[str, PlotElement] = {}
        self.elem_locations: Dict[str, Tuple[int, int]] = {}
        self.subplots: Dict[Tuple[int, int], List[PlotElement]] = {}
        self.subplot_titles: Dict[Tuple[int, int], str] = {}
        self.widget: go.FigureWidget

        self.last_drawn = None
        self._setup_figure()

    def _setup_figure(self):
        """Initialize the plotly figure"""
        
        specs = [[{"secondary_y": True} for _ in range(self.cols)] 
                 for _ in range(self.rows)]
        self.widget = go.FigureWidget(make_subplots(
            rows = self.rows,
            cols = self.cols,
            horizontal_spacing = 0.1,
            vertical_spacing = 0.1,
            specs = specs
        ))

        self.widget.update_layout(
            width = self.width,
            height = self.height,
            margin = {'l': self.width // 50, 'r': self.width // 50, 
                      't': self.height // 30, 'b': self.height // 30}
        )

    def show(self) -> go.FigureWidget:
        display(self.widget)
        return self.widget

    def draw(self, now = None):
        with self.widget.batch_update():
            for elem in self.elements.values():
                elem.draw(self.widget)
        self.last_drawn = now

    def update(self, data: Dict[str, dict]):
        for key, kwargs in data.items():
            self.elements[key].update(**kwargs)
        self._draw_lazy()


    def _draw_lazy(self):
        now = time.time()
        if self.last_drawn is None or \
            now - self.last_drawn > self.draw_interval:
            self.draw(now)
    
    def clear_element(self, key: str):
        with self.widget.batch_update():
            self.elements[key].clear(self.widget)

    def clear_subplot(self, row: int, col: int):
        with self.widget.batch_update():
            for elem in self.subplots[(row, col)]:
                elem.clear(self.widget)
    
    def clear(self):
        with self.widget.batch_update():
            for elem in self.elements.values():
                elem.clear(self.widget)

    def add_element(self, elem: PlotElement, row: int, col: int):
        if row > self.rows or col > self.cols or row < 1 or col < 1:
            raise ValueError(
                f"Subplot position ({row}, {col}) is out of bounds"
            )
        
        pos = (row, col)
        if pos not in self.subplots:
            self.subplots[pos] = []
        self.subplots[pos].append(elem)
        self.elements[elem.key] = elem
        self.elem_locations[elem.key] = pos

        elem.initialize(self.widget, row, col)

    def set_axis_labels(self, row: int, col: int, 
                        xlabel: str = None, 
                        ylabel: str = None, 
                        ylabel2: str = None,
                        elem_key: str = None):
        """Set axes labels for a specific subplot"""
        if elem_key:
            row, col = self.elem_locations[elem_key]

        if xlabel:
            self.widget.update_xaxes(row = row, col = col, title_text = xlabel)
        if ylabel:
            self.widget.update_yaxes(row = row, col = col, title_text = ylabel)
        if ylabel2:
            self.widget.update_yaxes(row = row, col = col, title_text = ylabel2,
                                     secondary_y = True)
            
    def set_axis_scale(self, row: int, col: int,
                       xtype: str = None,
                       ytype: str = None,
                       ytype2: str = None,
                       elem_key: str = None):
        """Set subplot axis type (linear, log, etc.)"""
        if elem_key:
            row, col = self.elem_locations[elem_key]

        if xtype:
            self.widget.update_xaxes(row = row, col = col, type = xtype)
        if ytype:
            self.widget.update_yaxes(row = row, col = col, type = ytype)
        if ytype2:
            self.widget.update_yaxes(row = row, col = col, type = ytype2,
                                     secondary_y = True)
            
    def set_axis_range(self, row: int, col: int,
                       xrange: Tuple[float | None, float | None] = None,
                       yrange: Tuple[float | None, float | None] = None,
                       yrange2: Tuple[float | None, float | None] = None,
                       elem_key: str = None):
        """Set subplot axis type (linear, log, etc.)"""
        if elem_key:
            row, col = self.elem_locations[elem_key]

        if xrange:
            self.widget.update_xaxes(row = row, col = col, range = xrange)
        if yrange:
            self.widget.update_yaxes(row = row, col = col, range = yrange)
        if yrange2:
            self.widget.update_yaxes(row = row, col = col, range = yrange2,
                                     secondary_y = True)

# Integration with param_sweep_measure and other common use cases

class SweepRecorder(Recorder):
    def __init__(self, 
                 variables: List[dict], 
                 line_kwargs: List[dict] = None,
                 twinx: List[Tuple[str, str]]  = None,
                 twin_axes: List[Tuple[str, ...]] = None, 
                 master_var: str = None,
                 max_cols: int = 5, 
                 width: int = None, height: int = None,
                 draw_interval: float = 0.2):
        
        twinx = twinx or []
        twin_axes = twin_axes or []
        xtwins = {**{y: o for o, y in twinx}}

        # takes a list like [('A', 'B', 'C'), ('D', 'E'), ...] to a dict like
        # {'A': ('B', 'C'), 'B': ('A', 'C'), ...}
        axtwins = {
            k: v for d in [
                {g[i]: (*g[:i], *g[i + 1:]) for i in range(len(g))} 
                for g in twin_axes
            ] 
            for k, v in d.items()
        }
        # I was so preoccupied with whether I could, I didn't stop to ask if I should...

        if master_var:
            for i, var in enumerate(variables):
                if var.get('name') == master_var:
                    self.master_var_ind = i
                    break
            else:
                raise ValueError(f"No variable with key: {master_var}")
            self.master_label = variables[i].get('long_name', master_var) + \
                ('' if not 'units' in var else f' [{var['units']}]')
            xscale = variables[i].get('scale')
        else:
            self.master_var_ind = None
            self.master_label = 'N'
            xscale = None

        nsub = 0
        seen = set()
        for v in variables:
            k = v.get('name')
            if k is None or k in [master_var, 'unnamed']:
                continue
            seen.add(k)
            if k in xtwins and xtwins[k] in seen:
                continue
            if k in axtwins:
                if any(t in seen for t in axtwins[k]):
                    continue
            nsub += 1

        cols = min(ceil(sqrt(nsub)), max_cols)
        rows = ceil(nsub / cols)
        width = min(1400, 400*rows)
        if height is None:
            height = int(0.6 * (width * rows) // cols)
        super().__init__(rows, cols, width, height, draw_interval)
        self.npoints = 0

        locations: Dict[str, Tuple[int, int]] = {}

        self.keys = {}
        line_kwargs = line_kwargs or [{}]*len(variables)
        pr = 1
        pc = 0
        for ii, (var, lkwargs) in enumerate(zip(variables, line_kwargs)):
            key = var.get('name')
            if key is None or key == 'unnamed':
                continue

            if key == master_var:
                continue
            
            self.keys[key] = ii
            r, c = None, None
            secondary_y = False
            younger_twin = False
            if key in xtwins:
                twin = xtwins[key]
                if twin in locations:
                    secondary_y = True
                    r, c = locations[twin]
            elif key in axtwins:
                younger_twin = True
                for twin in axtwins[key]:
                    if twin in locations:
                        r, c = locations[twin]
            if r is None or c is None:
                if pc == cols:
                    pc = 1
                    pr += 1
                else:
                    pc += 1
                r, c = pr, pc

            locations[key] = (r, c)
            label = var.get('long_name', key) + \
                ('' if not 'units' in var else f' [{var['units']}]')
            
            self.add_element(
                Line(
                    key = key,
                    name = label,
                    secondary_y = secondary_y,
                    legend = (key in axtwins),
                    **lkwargs
                ),
                row = r, col = c
            )

            label = var.get('units', '') if key in axtwins else label
            if not younger_twin:
                if secondary_y:
                    self.set_axis_labels(row = r, col = c,
                                         xlabel = self.master_label, 
                                         ylabel2 = label)
                    
                    self.set_axis_scale(row = r, col = c, 
                                        xtype = xscale,
                                        ytype2 = var.get('scale'))
                    
                else:
                    self.set_axis_labels(row = r, col = c,
                                         xlabel = self.master_label, 
                                         ylabel = label)
                    
                    self.set_axis_scale(row = r, col = c, 
                                        xtype = xscale,
                                        ytype = var.get('scale'))

    def update(self, coords: np.ndarray, data: np.ndarray = None):
        self.npoints += 1
        D = coords if data is None else np.hstack([coords, data])
        x = self.npoints if self.master_var_ind is None \
            else D[self.master_var_ind]
        for k, i in self.keys.items():
            self.elements[k].update(x, D[i])
        self._draw_lazy()

class ChartRecorder(Recorder):
    def __init__(self,):
        pass
