"""
Software representation of a HEMT amplifier with common-source topology.

                           VG       VDD
                            ┃        ┃
                         ╻┏━┛        ⌇ RDD
                   VM ━━━┫┃          ┣━━━┫┣━━ AC OUT  (VD monitors this node)
                         ╹┗━┓     ╻┏━┛
            AC IN ━━━┫┣━━━━━┻━━━━━┫┃
                                  ╹┗━┓
                                     ┃
                                     ⌇ RSS  (source branch, see
                                     ┃        circuit_elements.py)
                                    VS  <- the *rail*/return; driven or grounded

`VD` monitors the drain node -- between the drain branch (`RDD`) and the
transistor -- not the `VDD` rail itself. `VS` is the far end of the source
branch (the return rail; what the older, disabled hemt_amp.py called
VSS), grounded at 0V if not provided. Being SweepableChannel-typed lets VS
double as an alternate tuning axis alongside VG/VDD.

`VM`/`bias_transistor` are optional: some setups bias the gate node
through a *second* transistor pinched deep into its high-resistance
region rather than a physical resistor (a pinched-off transistor has
lower parasitic capacitance at that sensitive, high-impedance node than
an equivalent meander resistor would). `VM` drives that bias transistor's
own gate; `VG` acts as its "source" reference. Because essentially no DC
current flows through it (by design -- it's providing a bias path, not
carrying signal), it settles to negligible drop in steady state, so it
does not change the *main* transistor's own DC operating point at all --
`VG` still directly represents the main transistor's gate potential for
every calculation below. What it does add: its own hard safety window
(`VM - VG` must not go positive, for the same reason as the main
transistor -- see `bias_transistor`) and, empirically, real SNR
sensitivity to how hard it's pinched (more pinch generally raises the
resistance it presents at the gate node, which is why it usually helps --
but not monotonically, hence `VM` is a real search parameter for
`routines/amp_tune`, not a hardcoded offset from `VG` the way the older,
disabled `hemt_amp.py` treated it). If `VM`/`bias_transistor` are omitted,
`VG` is assumed to drive the gate node directly (no separate bias
element) -- the simpler diagram without the top-left branch.

The drain current I_D can come from either (or both) of two independent
sources, since the two together are enough to fully solve the operating
point:

    VD  : a voltage tap on the drain node, from which I_D is inferred via
          Ohm's law across the drain branch: I_D = (VDD - VD) / R_drain.
    IDS : a direct current-sense channel (e.g. a Keithley ammeter in
          series), read as-is with no dependence on R_drain at all.

If only one is provided, the other is derived from it (VD -> I_D via the
Ohm's-law relation above; I_D -> VD = VDD - I_D * R_drain). If both are
provided, IDS is treated as the authoritative current (a direct
measurement, not dependent on trusting R_drain's precise value) and VD is
used as-is for the drain node voltage -- so an inconsistency between the
two shows up as a discrepancy against the drain branch's *nominal*
R_drain, not as an error. At least one of VD/IDS must be provided;
otherwise ID()/VDS()/VGS()/dissipated_power_by_stage() raise -- see their
docstrings.

With I_D and the drain node voltage known, the physical source-pin
voltage follows from Ohm's law across the source branch:

    S_node = VS_rail + I_D * R_source
    VDS    = VD - S_node
    VGS    = VG() - S_node
"""

from __future__ import annotations

from .abstract_device import abstractDevice
from .channel_adapter import ScalarChannel
from .circuit_elements import (
    Branch,
    ResistiveElement,
    ThermalStage,
    TransistorElement,
    as_element_list,
    branch_power_by_stage,
    branch_resistance,
)
from .registry import DeferredReference, DeviceRegistry, format_reference, register_device_class
from .sweepable_channel import SweepableChannel

from logging import Logger
from typing import Any, Callable, Dict, Protocol, runtime_checkable


@runtime_checkable
class AmplifierLike(Protocol):
    """
    Structural interface the SNR-tuning framework (routines/amp_tune) needs
    from any amplifier topology. HEMTCommonSource is the first
    implementation; a future topology (cascode, differential pair, ...)
    only needs to satisfy this protocol to work with the same tuning
    routines unchanged, even though the circuit representation underneath
    stays a concrete series chain per topology.
    """
    def ID(self) -> float: ...
    def VDS(self) -> float: ...
    def VGS(self) -> float: ...
    def dissipated_power_by_stage(self) -> Dict[str, float]: ...
    def power_cost(self, weights: Dict[str, float] | None = None) -> float: ...
    def check_safe(self) -> bool: ...
    def move_to(self,
        panic_behavior: str | Callable = 'zero',
        min_wait: float | None = None,
        **targets: float,
    ) -> Any: ...
    def free_channels(self) -> Dict[str, SweepableChannel]: ...


@register_device_class("HEMTCommonSource")
class HEMTCommonSource(abstractDevice):
    """
    Common-source HEMT amplifier bias network.

    Parameters
    ----------
    VG : SweepableChannel
        Gate voltage.
    VDD : SweepableChannel
        Drain supply rail.
    drain : ResistiveElement or list of ResistiveElement
        Series resistor(s) between VDD and the drain node.
    transistor : TransistorElement
        The active device: thermal stage plus safe VGS/ID window.
    source : ResistiveElement, list of ResistiveElement, or None, default=None
        Series resistor(s) between the physical source pin and the VS
        rail. None means the source is directly grounded (0 Ohm).
    VD : ScalarChannel, optional
        Monitor of the drain node (between `drain` and the transistor).
        At least one of VD/IDS is required for ID()/VDS()/VGS()/
        dissipated_power_by_stage() -- see module docstring for why, and
        for how the two combine when both are provided.
    IDS : ScalarChannel, optional
        Direct current-sense reading of I_D (e.g. from an ammeter/SMU in
        series), used in place of -- or, if VD is also given, to
        corroborate -- the voltage-divider inference through `drain`.
    VS : SweepableChannel or ScalarChannel, optional
        The source branch's return rail. None means grounded (0V). If
        sweepable, this doubles as an alternate tuning axis alongside VG
        and VDD.
    VM : SweepableChannel, optional
        Gate voltage of the optional gate-bias transistor (used in place
        of a physical bias resistor -- see module docstring). Must be
        given together with `bias_transistor` or not at all. Always
        treated as a free tuning parameter (see `free_channels`), never
        just a monitor -- there is no useful "fixed VM" use case, unlike
        VS.
    bias_transistor : TransistorElement, optional
        The gate-bias transistor's own thermal stage and safe
        `VM - VG` window -- a physically distinct part from `transistor`,
        so it gets its own spec rather than reusing the main transistor's
        (their safe windows may coincide in practice, but nothing here
        assumes that). Not included in `dissipated_power_by_stage()` --
        it is deliberately operated near-pinch-off, carrying essentially
        no current, so its own dissipation is negligible by design.
    panic_home : dict, optional
        A known-safe bias point (keyed by 'VG'/'VDD'/'VS'/'VM') a caller
        can pass to move_to() to recover to if a live safety check trips.
        Purely advisory storage -- move_to()'s default panic_behavior
        ('zero') reverts to wherever the sweep started, which does not
        require this at all.
    registry_id : str, optional
        Name to register this instance under in the DeviceRegistry.
    logger : Logger, optional

    Notes
    -----
    Gate voltage is never silently clamped: the transistor model this was
    written for tolerates essentially no positive V_GS, so an unsafe
    target is refused (pre-flight, via check_safe()) rather than clamped
    to the edge of the allowed window. The same holds for the optional
    gate-bias transistor's VM - VG. move_to() additionally wires
    check_safe() into tandemSweep's live panic_condition, so an unsafe
    reading during the ramp itself reverts rather than being left in
    place -- defense in depth against a target that looked safe in
    advance but wasn't in practice.
    """

    def __init__(self,
        VG        : SweepableChannel,
        VDD       : SweepableChannel,
        drain     : Branch,
        transistor: TransistorElement,
        source    : Branch = None,
        VD        : ScalarChannel | None = None,
        IDS       : ScalarChannel | None = None,
        VS        : SweepableChannel | ScalarChannel | None = None,
        VM        : SweepableChannel | None = None,
        bias_transistor: TransistorElement | None = None,
        panic_home: Dict[str, float] | None = None,
        registry_id: str | None = None,
        logger    : Logger | None = None,
    ):
        super().__init__(logger)

        if (VM is None) != (bias_transistor is None):
            raise ValueError(
                "HEMTCommonSource: VM and bias_transistor must be provided "
                "together (a gate-bias transistor needs both its control "
                "channel and its own safe-operating-region spec) or both "
                "omitted."
            )

        self.VG  = VG
        self.VDD = VDD
        self.VD  = VD
        self.IDS = IDS
        self.VS  = VS
        self.VM  = VM

        self.drain           = drain
        self.source          = source
        self.transistor      = transistor
        self.bias_transistor = bias_transistor

        self.panic_home = panic_home

        self._warned_no_current_source = False

        if registry_id is not None:
            DeviceRegistry.register(self, registry_id=registry_id)

    """
    -------------------------------------------------------------------------
    Electrical model
    -------------------------------------------------------------------------
    """

    def _R_drain(self) -> float:
        return branch_resistance(self.drain)

    def _R_source(self) -> float:
        return branch_resistance(self.source)

    def _VS_rail(self) -> float:
        return self.VS() if self.VS is not None else 0.0

    def _require_operating_point(self, caller: str) -> None:
        if self.VD is None and self.IDS is None:
            raise ValueError(
                f"HEMTCommonSource.{caller}() requires at least one of VD "
                "or IDS; the operating point cannot be determined from "
                "VG/VDD alone."
            )

    def _read(self) -> Dict[str, float]:
        """
        Single hardware snapshot: reads VG/VDD plus whichever of VD/IDS
        are configured exactly once, and derives ID, VDS, and VGS from
        that one snapshot. ID()/VDS()/VGS() each take a fresh snapshot
        when called individually, but check_safe() and
        dissipated_power_by_stage() take exactly one snapshot internally
        and derive everything from it, rather than letting several
        internal accessor calls each re-query the same channels -- that
        would multiply hardware round-trips (real channels are often
        VISA queries) and risks an inconsistent reading if a channel
        drifts between calls, which matters most exactly when this is
        being used as a live safety check.

        If IDS is configured, it is authoritative for ID (a direct
        measurement, independent of R_drain); VD (if also configured) is
        then used as-read for the drain node voltage. If only VD is
        configured, ID is inferred from it via Ohm's law across the drain
        branch, which requires nonzero drain resistance. If only IDS is
        configured, the drain node voltage is instead inferred from it:
        VD = VDD - ID * R_drain.
        """
        self._require_operating_point('_read')
        VG, VDD = self.VG(), self.VDD()

        if self.IDS is not None:
            id_ = self.IDS()
            VD = self.VD() if self.VD is not None else VDD - id_ * self._R_drain()
        else:
            R_drain = self._R_drain()
            if R_drain <= 0:
                raise ValueError(
                    "HEMTCommonSource requires nonzero drain resistance to "
                    "infer current from VD alone; provide IDS for a direct "
                    "current reading instead."
                )
            VD = self.VD()
            id_ = (VDD - VD) / R_drain

        S_node = self._VS_rail() + id_ * self._R_source()
        return {'VG': VG, 'VDD': VDD, 'VD': VD, 'ID': id_,
                'VDS': VD - S_node, 'VGS': VG - S_node}

    def ID(self) -> float:
        """
        Drain current: read directly from IDS if configured, otherwise
        inferred via Ohm's law across the drain branch,
        ID = (VDD - VD) / R_drain. Requires at least one of VD/IDS -- see
        module docstring and _read().
        """
        return self._read()['ID']

    def VDS(self) -> float:
        """Drain-source voltage. Requires at least one of VD/IDS (see ID())."""
        return self._read()['VDS']

    def VGS(self) -> float:
        """Gate-source voltage. Requires at least one of VD/IDS (see ID())."""
        return self._read()['VGS']

    def VGS_bias(self) -> float:
        """
        Gate-source voltage of the optional gate-bias transistor:
        VM - VG, using VG as the "source" reference (see module
        docstring for why: it's the DC rail the bias transistor's
        channel connects to VG through). Independent of drain-current
        sensing -- unlike ID()/VDS()/VGS(), this works with neither VD
        nor IDS configured. Requires VM/bias_transistor.
        """
        if self.VM is None:
            raise ValueError(
                "HEMTCommonSource.VGS_bias() requires VM/bias_transistor "
                "to be configured."
            )
        return self.VM() - self.VG()

    """
    -------------------------------------------------------------------------
    Power accounting
    -------------------------------------------------------------------------
    """

    def _power_by_stage(self, id_: float, vds: float) -> Dict[str, float]:
        out = branch_power_by_stage(self.drain, id_)
        for stage_name, p in branch_power_by_stage(self.source, id_).items():
            out[stage_name] = out.get(stage_name, 0.0) + p

        t_stage = self.transistor.stage.name
        out[t_stage] = out.get(t_stage, 0.0) + self.transistor.power(id_, vds)
        return out

    def dissipated_power_by_stage(self) -> Dict[str, float]:
        """
        {stage_name: watts}, summing the transistor and every resistor
        element by the named thermal stage it was tagged with. Requires at
        least one of VD/IDS (see ID()).
        """
        op = self._read()
        return self._power_by_stage(op['ID'], op['VDS'])

    def power_cost(self, weights: Dict[str, float] | None = None) -> float:
        """
        Weighted sum of dissipated_power_by_stage(); the plain (unweighted)
        total if weights is None. Used both as a diagnostic and, by the
        SNR-tuning framework, as a tunable soft penalty term in the search
        objective.
        """
        by_stage = self.dissipated_power_by_stage()
        if weights is None:
            return sum(by_stage.values())
        return sum(w * by_stage.get(name, 0.0) for name, w in weights.items())

    """
    -------------------------------------------------------------------------
    Safety
    -------------------------------------------------------------------------
    """

    def _stages(self) -> Dict[str, ThermalStage]:
        stages = {self.transistor.stage.name: self.transistor.stage}
        for e in as_element_list(self.drain) + as_element_list(self.source):
            stages.setdefault(e.stage.name, e.stage)
        return stages

    def check_safe(self) -> bool:
        """
        True if the gate-bias transistor's VGS window (if configured),
        the main transistor's VGS/ID window, and every stage's max_power
        are currently satisfied.

        The gate-bias transistor check is independent of drain-current
        sensing (VGS_bias() only needs VM/VG) and so runs unconditionally
        whenever VM is configured -- it is a separate hardware read from
        the main _read() snapshot below (one extra VG query when both
        checks run) since _read() requires VD/IDS to be configured at
        all, which has nothing to do with the gate-bias transistor.

        The main transistor/power checks are evaluated from a single
        hardware snapshot (see _read()). If neither VD nor IDS is
        configured, that part of the operating point can't be evaluated
        at all -- a documented limitation (see the module docstring), so
        this returns True for it (nothing to check) after logging a
        one-time warning, rather than silently pretending the check
        passed with no record of why.
        """
        if self.VM is not None and not self.bias_transistor.is_safe(self.VGS_bias()):
            return False

        if self.VD is None and self.IDS is None:
            if not self._warned_no_current_source:
                self.warn(
                    "HEMTCommonSource.check_safe: neither VD nor IDS is "
                    "configured; electrical safety cannot be verified."
                )
                self._warned_no_current_source = True
            return True

        op = self._read()

        if not self.transistor.is_safe(op['VGS'], op['ID']):
            return False

        by_stage = self._power_by_stage(op['ID'], op['VDS'])
        for name, stage in self._stages().items():
            if stage.max_power is not None and by_stage.get(name, 0.0) > stage.max_power:
                return False

        return True

    """
    -------------------------------------------------------------------------
    Movement
    -------------------------------------------------------------------------
    """

    def free_channels(self) -> Dict[str, SweepableChannel]:
        """The bias channels that are actually sweepable: VG, VDD, VS if
        it was constructed as a SweepableChannel rather than a plain
        monitor or left grounded, and VM if a gate-bias transistor is
        configured (always sweepable when present -- there is no useful
        "fixed VM" case, unlike VS)."""
        channels = {'VG': self.VG, 'VDD': self.VDD}
        if isinstance(self.VS, SweepableChannel):
            channels['VS'] = self.VS
        if self.VM is not None:
            channels['VM'] = self.VM
        return channels

    def move_to(self,
        panic_behavior: str | Callable = 'zero',
        min_wait: float | None = None,
        **targets: float,
    ):
        """
        Move to a bias point, with check_safe() wired in as a live
        tandemSweep panic_condition. This is the sole entry point the
        SNR-tuning routines use to move the amp -- defense in depth
        against a candidate that a calibration surrogate mis-predicted as
        safe.

        Parameters
        ----------
        panic_behavior : str or Callable, default='zero'
            Passed through to tandemSweep. 'zero' reverts to wherever the
            sweep started if check_safe() ever fails during the ramp.
        min_wait : float, optional
            Passed through to tandemSweep.
        **targets : float
            Target values by channel name ('VG', 'VDD', 'VS', 'VM').
            Channels not named are held at their current value. Only
            channels returned by free_channels() may be targeted.

        Returns
        -------
        SweepResult
        """
        # Local import: devices/ intentionally has no top-level dependency
        # on core/ (core depends on devices, not the reverse); tandemSweep
        # is only needed here, at call time.
        from ..core.tandem_sweep import tandemSweep, SweepResult

        channels = self.free_channels()
        bad = set(targets) - set(channels)
        if bad:
            raise ValueError(f"Unknown/non-sweepable bias channel(s): {sorted(bad)}")

        names = list(channels)
        target = {name: targets.get(name, channels[name]()) for name in names}

        result = tandemSweep(
            channels        = [channels[name] for name in names],
            target          = target,
            min_wait        = min_wait,
            panic_condition = lambda _settings: not self.check_safe(),
            panic_behavior  = panic_behavior,
        )
        if result == SweepResult.PANICKED:
            self.warn("HEMTCommonSource.move_to: panicked; reverted.")
        return result

    """
    -------------------------------------------------------------------------
    Serialization
    -------------------------------------------------------------------------
    """

    def _serialize_state(self) -> Dict[str, Any]:
        return {
            'VG'        : format_reference(self.VG),
            'VDD'       : format_reference(self.VDD),
            'VD'        : format_reference(self.VD) if self.VD is not None else None,
            'IDS'       : format_reference(self.IDS) if self.IDS is not None else None,
            'VS'        : format_reference(self.VS) if self.VS is not None else None,
            'VM'        : format_reference(self.VM) if self.VM is not None else None,
            'drain'     : [e.to_dict() for e in as_element_list(self.drain)],
            'source'    : [e.to_dict() for e in as_element_list(self.source)],
            'transistor': self.transistor.to_dict(),
            'bias_transistor': self.bias_transistor.to_dict()
                               if self.bias_transistor is not None else None,
            'panic_home': self.panic_home,
        }

    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        if state.get('VG') is not None:
            self.VG = DeferredReference(state['VG']).unwrap()
        if state.get('VDD') is not None:
            self.VDD = DeferredReference(state['VDD']).unwrap()
        if state.get('VD') is not None:
            self.VD = DeferredReference(state['VD']).unwrap()
        if state.get('IDS') is not None:
            self.IDS = DeferredReference(state['IDS']).unwrap()
        if state.get('VS') is not None:
            self.VS = DeferredReference(state['VS']).unwrap()
        if state.get('VM') is not None:
            self.VM = DeferredReference(state['VM']).unwrap()
        if 'drain' in state:
            self.drain = [ResistiveElement.from_dict(d) for d in state['drain']]
        if 'source' in state:
            self.source = [ResistiveElement.from_dict(d) for d in state['source']] or None
        if 'transistor' in state and state['transistor'] is not None:
            self.transistor = TransistorElement.from_dict(state['transistor'])
        if state.get('bias_transistor') is not None:
            self.bias_transistor = TransistorElement.from_dict(state['bias_transistor'])
        if 'panic_home' in state:
            self.panic_home = state['panic_home']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HEMTCommonSource':
        registry_id = config.pop('registry_id', None)

        VG_ref  = config.pop('VG')
        VDD_ref = config.pop('VDD')
        VD_ref  = config.pop('VD', None)
        IDS_ref = config.pop('IDS', None)
        VS_ref  = config.pop('VS', None)
        VM_ref  = config.pop('VM', None)

        drain      = [ResistiveElement.from_dict(d) for d in config.pop('drain', [])]
        source     = [ResistiveElement.from_dict(d) for d in config.pop('source', [])]
        transistor = TransistorElement.from_dict(config.pop('transistor'))
        bias_transistor_dict = config.pop('bias_transistor', None)

        instance = cls(
            VG          = None,  # resolved in _deserialize_state (pass 2)
            VDD         = None,
            drain       = drain,
            transistor  = transistor,
            source      = source or None,
            VD          = None,
            IDS         = None,
            VS          = None,
            VM          = None,  # both VM and bias_transistor deferred below,
            bias_transistor = None,  # to dodge __init__'s "both or neither" check
            panic_home  = config.pop('panic_home', None),
            registry_id = registry_id,
        )
        instance.VG  = DeferredReference(VG_ref)
        instance.VDD = DeferredReference(VDD_ref)
        if VD_ref is not None:
            instance.VD = DeferredReference(VD_ref)
        if IDS_ref is not None:
            instance.IDS = DeferredReference(IDS_ref)
        if VS_ref is not None:
            instance.VS = DeferredReference(VS_ref)
        if VM_ref is not None:
            instance.VM = DeferredReference(VM_ref)
        if bias_transistor_dict is not None:
            instance.bias_transistor = TransistorElement.from_dict(bias_transistor_dict)

        return instance


if __name__ == '__main__':
    """
    Self-test using plain in-memory fake channels (no VISA hardware is
    involved anywhere in HEMTCommonSource, so there is no dummyResource
    equivalent needed here -- unlike the pyvisaDevice-based examples
    elsewhere in this package).
    """

    from .sweepable_channel import SweepConfig

    class FakeChannel:
        """Minimal ScalarChannel: a mutable float box."""
        def __init__(self, value: float = 0.0):
            self._v = value
        def get_output(self) -> float:
            return self._v
        def set_output(self, v: float) -> None:
            self._v = v
        def __call__(self, v: float | None = None) -> float | None:
            if v is None:
                return self.get_output()
            self.set_output(v)

    def check(label: str, cond: bool):
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        assert cond, label

    stage_1k = ThermalStage('1K', max_power=50e-6)
    stage_mc = ThermalStage('MC', max_power=200e-6)
    transistor = TransistorElement(stage=stage_mc, vgs_min=-0.6, vgs_max=0.0)
    drain = ResistiveElement(resistance=1000.0, stage=stage_1k)

    VG_raw, VDD_raw, VD_raw = FakeChannel(-0.3), FakeChannel(1.0), FakeChannel(0.9)
    cfg = SweepConfig(settle_time=0.001, max_step=0.05, tolerance=1e-6)
    VG  = SweepableChannel(VG_raw,  config=cfg, name='VG')
    VDD = SweepableChannel(VDD_raw, config=cfg, name='VDD')

    amp = HEMTCommonSource(
        VG=VG, VDD=VDD, drain=drain, transistor=transistor, source=None, VD=VD_raw,
    )

    ID = amp.ID()
    check('ID == (VDD - VD) / R_drain', abs(ID - (1.0 - 0.9) / 1000.0) < 1e-12)
    check('VDS == VD for a grounded source', abs(amp.VDS() - 0.9) < 1e-12)
    check('VGS == VG for a grounded source', abs(amp.VGS() - (-0.3)) < 1e-12)
    check('check_safe() True at a nominal bias point', amp.check_safe() is True)

    VG_raw.set_output(0.1)
    check('check_safe() False for a positive VGS', amp.check_safe() is False)
    VG_raw.set_output(-0.3)

    print("MOVING TO VG = -0.2 (safe)")
    from ..core.tandem_sweep import SweepResult
    result = amp.move_to(VG=-0.2)
    check('move_to SUCCEEDED for a safe target', result == SweepResult.SUCCEEDED)

    print("MOVING TO VG = 0.5 (crosses into the unsafe VGS>0 region)")
    result = amp.move_to(VG=0.5)
    check('move_to PANICKED and reverted for an unsafe target',
          result == SweepResult.PANICKED and abs(VG_raw.get_output() - (-0.2)) < 1e-6)

    # A direct current-sense channel (e.g. a Keithley ammeter in series)
    # works in place of -- or, if VD is also given, alongside -- the
    # voltage-divider inference through the drain branch.
    IDS_raw = FakeChannel(4e-4)
    amp_ids = HEMTCommonSource(
        VG=VG, VDD=VDD, drain=drain, transistor=transistor, source=None,
        VD=None, IDS=IDS_raw,
    )
    check('ID() reads directly from IDS when VD is absent',
          abs(amp_ids.ID() - 4e-4) < 1e-15)

    # -- Gate-bias transistor (VM) ---------------------------------------
    try:
        HEMTCommonSource(
            VG=VG, VDD=VDD, drain=drain, transistor=transistor, VD=VD_raw,
            VM=SweepableChannel(FakeChannel(-0.9), config=cfg, name='VM'),
        )
        check('VM without bias_transistor raises', False)
    except ValueError:
        check('VM without bias_transistor raises', True)

    bias_transistor = TransistorElement(stage=stage_mc, vgs_min=-1.0, vgs_max=0.0)
    VM_raw = FakeChannel(-0.9)
    VM = SweepableChannel(VM_raw, config=cfg, name='VM')
    amp_vm = HEMTCommonSource(
        VG=VG, VDD=VDD, drain=drain, transistor=transistor, source=None,
        VD=VD_raw, VM=VM, bias_transistor=bias_transistor,
    )
    check('VGS_bias == VM - VG', abs(amp_vm.VGS_bias() - (-0.9 - VG())) < 1e-12)
    check("'VM' is a free channel", 'VM' in amp_vm.free_channels())
    check('check_safe() True with VM safely pinched off', amp_vm.check_safe() is True)

    VM_raw.set_output(0.1)  # VM - VG = 0.1 - VG() > 0 (VG is <= 0): unsafe
    check('check_safe() False when VM - VG goes positive', amp_vm.check_safe() is False)
    VM_raw.set_output(-0.9)

    print("MOVING VM more positive than VG (crosses into VM-VG>0)")
    result = amp_vm.move_to(VM=0.1)
    check('move_to PANICKED and reverted for an unsafe VM target',
          result == SweepResult.PANICKED and abs(VM_raw.get_output() - (-0.9)) < 1e-6)

    # Round-trip through the registry to check VM/bias_transistor survive
    # serialization (including the "both deferred, dodge __init__'s
    # validation" path in from_config) -- needs actually-registered
    # channels, unlike the plain FakeChannels used above.
    from .registry import DeviceRegistry
    VG2  = SweepableChannel(FakeChannel(-0.3), config=cfg, name='VG',  registry_id='test_VG2')
    VDD2 = SweepableChannel(FakeChannel(1.0),  config=cfg, name='VDD', registry_id='test_VDD2')
    VD2  = SweepableChannel(FakeChannel(0.9),  config=cfg, name='VD',  registry_id='test_VD2')
    VM2  = SweepableChannel(FakeChannel(-0.9), config=cfg, name='VM',  registry_id='test_VM2')
    amp_vm_reg = HEMTCommonSource(
        VG=VG2, VDD=VDD2, drain=drain, transistor=transistor, source=None,
        VD=VD2, VM=VM2, bias_transistor=bias_transistor,
    )
    state = amp_vm_reg._serialize_state()
    amp_vm_reg2 = HEMTCommonSource.from_config({**state, 'registry_id': 'test_amp_vm2'})
    amp_vm_reg2._deserialize_state(state)
    check('deserialized amp resolved VM to the same registered channel',
          amp_vm_reg2.VM is VM2)
    check('deserialized amp kept the bias transistor safe window',
          amp_vm_reg2.bias_transistor.vgs_min == -1.0)
    DeviceRegistry.clear()

    print("\nAll checks passed.")
