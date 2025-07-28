"""
Usage:

# Cell [1]
%load_ext pyPulses.thread_cell_magic

# Cell [2]
%%thread_cell_magic

<relevant code>

This is a lazy way of interpreting the contents of an IPython notebook cell as a
ThreadJob and launching it automatically. I heavily discourage overuse of this,
because it promotes bad practice and can lead to confusion with immutable types
due to scope confusion.

eg:

# Cell [1]
x = 0

# Cell [2]
%%threadjob
x = 1

# Cell [3]
print(f"x = {x}")

will yield x = 0, because the contents of cell [2] gets wrapped in a function
that has local scope.

Setting aside such potential confusions, it is generally better practice to
define threaded procedures as functions since we are liable to repeat them over
the course of measuring a sample.
"""

import textwrap
import traceback
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

from .thread_job import ThreadJob, auto_checkpoint

_counter = 0  # Ensures unique function/job names across invocations
_last_temp_threadjob: ThreadJob | None = None

@register_cell_magic
def threadjob(line, cell):
    """
    Usage:
        %%threadjob
        <your threaded code>

    Optional arguments:
        nocontrol   - disables display of pause/resume/stop controls
        noinjection - disables application of auto_checkpoint to provided code
    """
    global _counter
    global _last_temp_threadjob 

    if _last_temp_threadjob and _last_temp_threadjob.is_alive():
        print("[threadjob] Stopping prior cell job...")
        _last_temp_threadjob.stop()
        _last_temp_threadjob.join(timeout=5)

    ip = get_ipython()
    user_ns = ip.user_ns
    idx = _counter
    _counter += 1

    # Generate unique names
    func_name = f'_threadjob_func_{idx}'
    job_name = f'_threadjob_job_{idx}'
    cleanup_name = f'_threadjob_cleanup_{idx}'

    # Build the function definition
    func_code = f"def {func_name}():\n{textwrap.indent(cell, '    ')}"
    try:
        exec(func_code, user_ns)
        if 'noinjection' not in line:
            try:
                user_ns[func_name] = auto_checkpoint(user_ns[func_name], 
                                                     src = func_code)
            except Exception:
                print("[threadjob] Warning: failed to apply " \
                      "auto_checkpoint to cell function.")
    except Exception:
        print("[threadjob] Error compiling function:")
        traceback.print_exc()
        return
    
    # Create the job
    job = ThreadJob(user_ns[func_name])
    user_ns[job_name] = job

    # Define cleanup function
    def _cleanup(*_):
        for name in [func_name, job_name, cleanup_name]:
            user_ns.pop(name, None)

    user_ns[cleanup_name] = _cleanup
    job.on('finish', _cleanup)
    job.on('error', _cleanup)
    job.on('stop', _cleanup)

    # Optional control widget
    if 'nocontrol' not in line:
        try:
            job.show_controls()
        except Exception:
            print("[threadjob] Failed to show controls")

    # Start the job
    try:
        job.start()
    except Exception:
        print("[threadjob] Failed to start job:")
        traceback.print_exc()

    _last_temp_threadjob = job

def load_ipython_extension(ipython):
    """This makes thread_cell_magic.py a valid IPython extension."""
    pass
