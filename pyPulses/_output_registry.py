import uuid
import time
import threading
import weakref
from typing import Dict, List
from IPython import get_ipython
from IPython.display import display, clear_output
import ipywidgets as widgets

import logging
_logger = logging.getLogger(__name__)

_lock = threading.RLock()

class OutputRegistry:
    # weak maps: id -> widget (not owning)
    _outputs: 'weakref.WeakValueDictionary[str, widgets.Output]' = weakref.WeakValueDictionary()
    _panels:  'weakref.WeakValueDictionary[str, widgets.Widget]' = weakref.WeakValueDictionary()

    # persistent metadata (label, created_at, type, job_id, keep_on_finish)
    _meta: Dict[str, dict] = {}

    # strong pins: id -> strong_ref (only when owner explicitly pins)
    _pins: Dict[str, object] = {}

    @classmethod
    def new_output(cls, label: str = None, auto_display: bool = False):
        with _lock:
            oid = str(uuid.uuid4())
            out = widgets.Output()
            cls._outputs[oid] = out
            cls._meta[oid] = {
                "label": label,
                "created_at": time.time(),
                "type": "output",
                "job_id": None,
                "keep_on_finish": True,
            }
            if auto_display:
                display(out)
            return oid, out

    @classmethod
    def register_panel(cls, panel_widget: widgets.Widget, label: str = None,
                       out_id: str = None, job_id: str = None,
                       keep_on_finish: bool = True) -> str:
        with _lock:
            pid = str(uuid.uuid4())
            cls._panels[pid] = panel_widget
            cls._meta[pid] = {
                "label": label,
                "created_at": time.time(),
                "type": "panel",
                "job_id": job_id,
                "out_id": out_id,
                "keep_on_finish": keep_on_finish,
            }
            # display the panel once; do not attempt to display inside Output
            display(panel_widget)
            return pid

    @classmethod
    def attach_job(cls, id: str, job_id: str):
        with _lock:
            if id in cls._meta:
                cls._meta[id]["job_id"] = job_id

    @classmethod
    def get_output(cls, oid: str) -> widgets.Output | None:
        return cls._outputs.get(oid)

    @classmethod
    def get_panel(cls, pid: str) -> widgets.Widget | None:
        return cls._panels.get(pid)

    @classmethod
    def pin(cls, id: str, obj: object):
        """
        Ensure a strong reference to `obj` is kept while pinned (owner provides 
        obj).
        """
        with _lock:
            cls._pins[id] = obj

    @classmethod
    def unpin(cls, id: str):
        with _lock:
            cls._pins.pop(id, None)

    @classmethod
    def redisplay(cls, id: str):
        """
        Re-display the widget (if still alive) into its Output or directly.
        Returns True if re-displayed, False otherwise.
        """
        with _lock:
            meta = cls._meta.get(id)
            if not meta:
                return False

            if meta.get('type') == 'output':
                out = cls._outputs.get(id)
                if out is None:
                    return False
                # clear previous content and display the widget(s) again
                # we expect owner to have stored what to display inside Output originally
                # Common pattern: Output contains the FigureWidget or a panel.
                with out:
                    clear_output(wait=True)
                    # attempt to find a "primary" child widget to re-display:
                    # some owners store the widget in meta['primary_widget_ref'] if needed.
                    primary = meta.get('primary_widget')
                    if primary is not None:
                        display(primary)
                    else:
                        # fallback: display nothing (owner should have set 'primary_widget')
                        pass
                return True

            elif meta.get('type') == 'panel':
                panel = cls._panels.get(id)
                if panel is None:
                    return False
                # display the panel directly (it will create/restore a view)
                display(panel)
                return True

            return False

    @classmethod
    def clear(cls, id: str, remove_display: bool = False) -> bool:
        with _lock:
            meta = cls._meta.get(id)
            if not meta:
                return False

            typ = meta.get('type')
            if typ == 'output':
                out = cls._outputs.pop(id, None)
                cls._meta.pop(id, None)
                # clear output area
                if out is not None:
                    try:
                        with out:
                            clear_output(wait=True)
                        if remove_display:
                            out.close()

                        # best-effort close comms for any primary_widget
                        primary = cls._meta.get(id, {}).get('primary_widget')
                        if primary is not None:
                            cls._close_comm_for_widget(primary)
                    except Exception:
                        pass
                # drop any strong pin
                cls._pins.pop(id, None)
                return True

            elif typ == 'panel':
                panel = cls._panels.pop(id, None)
                cls._meta.pop(id, None)
                if panel is not None:
                    try:
                        if remove_display and hasattr(panel, 'close'):
                            panel.close()
                        # try to close comms for the panel object
                        cls._close_comm_for_widget(panel)
                    except Exception:
                        pass
                cls._pins.pop(id, None)
                return True

            return False

    @classmethod
    def list(cls) -> Dict[str, dict]:
        with _lock:
            # return shallow copy of metadata (don't resurrect dead weakrefs)
            return {k: dict(v) for k, v in cls._meta.items()}

    @classmethod
    def find_by_job(cls, job_id: str) -> List[str]:
        with _lock:
            return [k for k, v in cls._meta.items() if v.get('job_id') == job_id]

    @classmethod
    def clear_by_label(cls, label: str, remove_display: bool = True):
        with _lock:
            targets = [k for k, v in cls._meta.items() if v.get('label') == label]
        cnt = 0
        for tid in targets:
            if cls.clear(tid, remove_display=remove_display):
                cnt += 1
        return cnt

    @classmethod
    def list_active(cls):
        """
        List all currently registered outputs/panels with age and metadata.
        """
        now = time.time()
        entries = []
        for k, v in cls._meta.items():
            age = now - v.get("created_at", now)
            entries.append({
                "id": k,
                "label": v.get("label"),
                "type": v.get("type"),
                "job_id": v.get("job_id"),
                "keep_on_finish": v.get("keep_on_finish"),
                "age_sec": age,
            })
        return entries

    @classmethod
    def dismiss_by_label(cls, label: str, remove_display: bool = True):
        """
        Unregister and clear all widgets with a given label.
        Returns the number cleared.
        """
        targets = [k for k, v in cls._meta.items() if v.get("label") == label]
        for tid in targets:
            try:
                cls.clear(tid, remove_display=remove_display)
            except Exception:
                pass
        return len(targets)

    @classmethod
    def dismiss_older_than(cls, seconds: float, remove_display: bool = True):
        """
        Unregister and clear all widgets older than `seconds`.
        Returns the number cleared.
        """
        now = time.time()
        targets = [
            k for k, v in cls._meta.items()
            if (now - v.get("created_at", now)) > seconds
        ]
        for tid in targets:
            try:
                cls.clear(tid, remove_display=remove_display)
            except Exception:
                pass
        return len(targets)

    @classmethod
    def dismiss_all(cls, remove_display: bool = True):
        """
        Clear and unregister all outputs and panels.
        Returns the number cleared.
        """
        targets = list(cls._meta.keys())
        for tid in targets:
            try:
                cls.clear(tid, remove_display=remove_display)
            except Exception:
                pass
        return len(targets)

    # Automatic Notebook Pruning

    _prune_thread = None
    _prune_stop_event = None
    _prune_interval_sec = 24 * 3600
    _prune_cutoff_sec = 24 * 3600
    _prune_check_interval = 60  # check every 60s for stop or config changes

    @classmethod
    def _auto_prune_loop(cls):
        """Background thread that prunes at a configurable interval."""
        last_prune = 0
        while not cls._prune_stop_event.is_set():
            now = time.time()

            # If enough time has elapsed, run prune
            if now - last_prune >= cls._prune_interval_sec:
                try:
                    cleared = cls.dismiss_older_than(
                        cls._prune_cutoff_sec, remove_display=True
                    )
                    if cleared:
                        _logger.info(
                            "OutputRegistry: auto-pruned %d stale widgets", cleared
                        )
                except Exception:
                    _logger.exception("OutputRegistry: error in auto-prune")

                last_prune = now

            # Sleep in short increments to allow early reconfig or disable
            cls._prune_stop_event.wait(timeout=cls._prune_check_interval)

        _logger.debug("OutputRegistry: auto-prune loop exited")

    @classmethod
    def enable_auto_prune(cls, every_hours: float = 24.0, older_than_hours: float = 24.0):
        """Enable (or reconfigure) background auto-prune thread."""
        cls._prune_interval_sec = float(every_hours) * 3600.0
        cls._prune_cutoff_sec = float(older_than_hours) * 3600.0

        if cls._prune_thread and cls._prune_thread.is_alive():
            _logger.info(
                "OutputRegistry: auto-prune interval updated "
                "(every %.2fh, older_than %.2fh)",
                every_hours, older_than_hours,
            )
            return

        cls._prune_stop_event = threading.Event()
        cls._prune_thread = threading.Thread(
            target=cls._auto_prune_loop,
            daemon=True,
            name="OutputRegistryAutoPrune"
        )
        cls._prune_thread.start()
        _logger.info(
            "OutputRegistry: auto-prune enabled (every %.2fh, older_than %.2fh)",
            every_hours, older_than_hours,
        )

    @classmethod
    def disable_auto_prune(cls):
        """Stop the background prune thread."""
        if cls._prune_stop_event:
            cls._prune_stop_event.set()
            _logger.info("OutputRegistry: auto-prune disable requested")
        else:
            _logger.debug("OutputRegistry: auto-prune not active")


    # Best effort attempt to close comms associated with a widget

    @classmethod
    def _close_comm_for_widget(cls, widget):
        """
        Best-effort close of the kernel comm associated with a widget instance.
        Many ipywidgets subclasses expose .comm (a Comm object). If present, we
        try to close the kernel-side comm object (comm_manager).
        This reduces lingering comms visible in kernel.comm_manager.comms.

        This is safe but conservative: if the widget has no .comm or the comm id
        is not present in the kernel manager, nothing happens.
        """
        try:
            if widget is None:
                return False

            # many widgets inherit from ipywidgets.widgets.widget.DOMWidget
            # which exposes .comm; some (Output) may not directly expose a .comm
            comm = getattr(widget, 'comm', None)
            if comm is None:
                # Try to detect nested children that have comms (e.g., Output may
                # contain inner widgets). We don't recurse deeply — just one level.
                try:
                    children = getattr(widget, 'children', None)
                    if children:
                        for ch in children:
                            c = getattr(ch, 'comm', None)
                            if c is not None:
                                comm = c
                                break
                except Exception:
                    comm = None

            if comm is None:
                return False

            comm_id = getattr(comm, 'comm_id', None)
            if not comm_id:
                return False

            km = get_ipython().kernel.comm_manager
            # if manager has a comm with that id, close it
            if comm_id in km.comms:
                try:
                    km.comms[comm_id].close()
                except Exception:
                    # swallow errors: closing is best-effort
                    pass
                # also remove from manager mapping if present
                try:
                    km.comms.pop(comm_id, None)
                except Exception:
                    pass
                return True
            else:
                return False

        except Exception:
            return False
