import ast
import inspect
import textwrap
import threading
import time
import traceback
from contextvars import ContextVar
from types import ModuleType
from typing import List


"""Thread-local / current-control plumbing"""

class StopRequested(Exception):
    pass

class Control():
    def __init__(self):
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._pause.clear()
        self._stop.clear()

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop.set()

    def wait_if_paused_or_stopped(self):
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.1)
        if self._stop.is_set():
            raise StopRequested()

# context-var holding the current Control for running the job thread
_current_control: ContextVar[Control | None] = \
    ContextVar('_current_control', default = None)

def _checkpoint():
    """Cooperatively pauses / stops if a Control is active."""
    cntrl = _current_control.get()
    if cntrl is None:
        return # no-op outside jobs
    cntrl.wait_if_paused_or_stopped()

"""AST transformer + decorator to inject _checkpoint into loops"""

class _CheckpointInjector(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        node.body.insert(0, _make_checkpoint_call())
        return node
    
    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        node.body.insert(0, _make_checkpoint_call())
        return node
    
def _make_checkpoint_call():
    return ast.Expr(
        value = ast.Call(
            func = ast.Name(id = '_checkpoint', ctx = ast.Load()),
            args = [], keywords = []
        )
    )

def auto_checkpoint(func):
    """
    Decorate a function to inject `_checkpoint()` at the start of each for/while
    loop. If source can't be obtained, it silently falls back to the original
    function.
    """

    try:
        src = inspect.getsource(func)
    except (OSError, IOError, TypeError):
        # couldn't get source (built-in, C extensions, etc.)
        return func
    
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # find the function def node
    target = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
            node.name == func.__name__:
            target = node
            break
    
    if target is None:
        # couldn't find the def node
        return func
    
    # sanitize the decorator_list
    # This is necessary in order to avoid infinite recursion where it tries to
    # apply the decorator over and over again
    target.decorator_list = [
        dec for dec in target.decorator_list
        if not (
            isinstance(dec, ast.Name) and dec.id == 'auto_checkpoint'
        )
    ]

    # inject checkpoints
    injector = _CheckpointInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)

    # rebuild function with same globals, ensuring _checkpoint is in scope
    g = dict(func.__globals__)
    g['_checkpoint'] = _checkpoint

    code = compile(tree, inspect.getsourcefile(func) or '<string>', 'exec')
    l: dict[str, object] = {}
    exec(code, g, l)

    new_func = l.get(func.__name__)
    return new_func if callable(new_func) else func

def patch_functions(module: ModuleType, names: List[str]):
    """
    Monkey-patch functions in an already-imported module to inject checkpoints.
    Example
    -------
    import mypkg.submod as sm
    patch_functions(sm, ['long_loop_a', 'long_loop_b'])
    """

    for name in names:
        f = getattr(module, name, None)
        if callable(f):
            setattr(module, name, auto_checkpoint(f))

"""ThreadJob: run in same process"""
class ThreadJob:
    def __init__(self, func, *args, **kwargs):
        """
        func(*args, **kwargs) will run in the background thread with a Control
        available via the contextvar (used by `_checkpoint()`).
        """

        self.func = func
        self.args = args
        self.kwargs = kwargs

        self.control = Control()
        self.thread: threading.Thread | None = None
        self.exc: BaseException | None = None
        self.result = None

        # Public user hooks
        self.on_start = None
        self.on_stop = None
        self.on_finish = None
        self.on_error = None

        # Internal event listeners
        self._events = {'start': [], 'stop': [], 'finish': [], 'error': []}

    def on(self, event, callback):
        """Register internal listener for 'start', 'stop', 'finish', 'error'."""
        if event not in self._events:
            raise ValueError(f"Unknown event: {event}")
        self._events[event].append(callback)

    def _emit(self, event, *args):
        for cb in self._events.get(event, []):
            try:
                cb(self, *args)
            except Exception:
                traceback.print_exc()

    def _runner(self):
        token = _current_control.set(self.control)
        try:
            self._emit('start')
            if callable(self.on_start):
                try:
                    self.on_start(self)
                except Exception:
                    traceback.print_exc()

            self.result = self.func(*self.args, **self.kwargs)

            self._emit('finish', self.result)
            if callable(self.on_finish):
                try:
                    self.on_finish(self, self.result)
                except Exception:
                    traceback.print_exc()

        except StopRequested:
            self._emit('stop', None)
            if callable(self.on_stop):
                try:
                    self.on_stop(self, None)
                except Exception:
                    traceback.print_exc()

        except Exception as e:
            self.exc = e
            tb = traceback.format_exc()
            self._emit('error', tb)
            if callable(self.on_error):
                try:
                    self.on_error(self, tb)
                except Exception:
                    traceback.print_exc()

        finally:
            _current_control.reset(token)

    def start(self):
        if self.thread and self.thread.is_alive():
            print("Thread already running.")
            return
        self.thread = threading.Thread(target=self._runner, daemon=True)
        self.thread.start()

    def pause(self):
        self.control.pause()

    def resume(self):
        self.control.resume()

    def stop(self):
        self.control.stop()

    def is_alive(self):
        return self.thread and self.thread.is_alive()

    def show_controls(self):
        """Display pause/resume/stop buttons inline (only in Jupyter)."""
        if not _in_notebook():
            print("Control UI only works in Jupyter notebooks.")
            return

        import ipywidgets as widgets
        from IPython.display import display

        button_layout = widgets.Layout(width='50px')
        pause_btn = widgets.Button(description="⏸", button_style='warning',
                                   tooltip="Pause", layout=button_layout)
        resume_btn = widgets.Button(description="▶", button_style='success',
                                    tooltip="Resume", layout=button_layout)
        stop_btn = widgets.Button(description="⏹", button_style='danger',
                                  tooltip="Stop", layout=button_layout)

        status_label = widgets.HTML(value="")
        buttons = widgets.HBox([pause_btn, resume_btn, stop_btn])
        panel = widgets.VBox([buttons, status_label])
        display(panel)

        def pause_clicked(_): self.pause()
        def resume_clicked(_): self.resume()
        def stop_clicked(_): self.stop()

        pause_btn.on_click(pause_clicked)
        resume_btn.on_click(resume_clicked)
        stop_btn.on_click(stop_clicked)

        def finished_cleanup(job, *args):
            panel.children = [widgets.HTML(value="<b>Job finished</b>")]
        def stopped_cleanup(job, *args):
            panel.children = [widgets.HTML(value="<b>Job stopped</b>")]
        def show_error(job, tb):
            panel.children = [widgets.HTML(
                value=f"<b>Error</b><br><pre style='font-size:smaller'>{tb}</pre>"
            )]

        self.on('finish', finished_cleanup)
        self.on('stop', stopped_cleanup)
        self.on('error', show_error)

    def start_with_controls(self):
        self.show_controls()
        self.start()


"""Check to see if we're in a notebook"""
def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return "ZMQInteractiveShell" in shell
    except Exception:
        return False
