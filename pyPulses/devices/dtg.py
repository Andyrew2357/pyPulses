"""
Instrument control for 5000 series Tektronix data timing generators.
"""

from .pyvisa_device import pyvisaDevice
from .dtg_utils import *

import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray
from typing import Callable, Dict, List

"""Base DTG class"""

class DTG(pyvisaDevice):
    """Base DTG control class for the 5000 series data timing generators"""
    def __init__(self, logger = None, instrument_id: str = None):
        
        assert hasattr(self, 'pyvisa_config')
        assert hasattr(self, 'MODULES')
        assert hasattr(self, 'mainframes')
        assert hasattr(self, 'slots')

        super().__init__(self.pyvisa_config, logger, instrument_id)

        self.modules    : Dict[str, int]     = {}
        self.channels   : Dict[str, Channel] = {}
        self.groups     : Dict[str, Group]   = {}
        self.blocks     : Dict[str, Block]   = {}
        self.sequence   : List[str]          = []
        self.mode           : str   = None
        self._frequency     : float = None
        self._time_offset   : float = None
        self._burst_count   : int   = None

        self._get_installed_modules()
        self.operation_mode()
    
    def _mode_required(*allowed_modes) -> Callable:
        """Check to see if we are in the right mode to call this function."""

        def decorator(f: Callable) -> Callable:
            def wrapper(self, *args, **kwargs):
                if self.mode not in allowed_modes:
                    self.error(
                        f"Cannot call {f.__name__} in {self.mode}\n"
                        f"Allowed modes are: {allowed_modes}"
                    )
                    return
                return f(self, *args, **kwargs)
            return wrapper
        return decorator

    def _get_installed_modules(self) -> dict:
        """Determine installed hardware modules"""

        for mf in self.mainframes:
            for slot in self.slots:
                module_id = int(self.query(f"PGEN{slot}{mf}:ID?"))
                self.modules[(slot, mf)] = module_id
                if module_id != -1:
                    for ch in range(1, self.MODULES[module_id]['n_ch'] + 1):
                        self.channels[f"{mf}{slot}{ch}"] = Channel(
                            name        = f"{mf}{slot}{ch}",
                            ch          = ch,
                            slot        = slot,
                            mf          = mf,
                            min_V       = self.MODULES[module_id]['min_V'],
                            max_V       = self.MODULES[module_id]['max_V'],
                            min_V_diff  = self.MODULES[module_id]['min_V_diff'],
                            min_width   = self.MODULES[module_id]['min_width'],
                            rise_time   = self.MODULES[module_id]['rise_time'],
                            dtg         = self,
                        )

        return self.modules
    
    def get_channel(self, ch: str | Channel) -> Channel | None:
        """
        Get a channel by its string id (if it exists).

        Parameters
        ----------
        ch : str
            channel name.
        
        Returns
        -------
        channel : Channel
        """
        if isinstance(ch, Channel):
            return ch
        
        if not ch in self.channels:
            self.error(f"Channel '{ch}' does not exist.")
            return
        
        return self.channels[ch]
    
    def operation_mode(self, pg: bool = None) -> bool | None:
        """
        Set or query the operation mode.
        
        Parameters
        ----------
        pg : bool, optional
            true = pulse generator mode, false = data generator mode.
        
        Returns
        -------
        operation_mode : bool or None
            only returns if `pg` is None (query mode).
        """

        if pg is None:
            self.mode = self.query("TBAS:OMODE?").strip()
            return self.mode == 'PULS'
        
        self.mode = 'PULS' if pg else 'DATA'
        self.write(f"TBAS:OMODE {self.mode}")
        self.info(f"Set operational mode to {'pulse' if pg else 'data'} mode.")

    def burst_mode(self, burst: bool = None) -> bool | None:
        """
        Set or query whether we are in burst or continuous mode.
        
        Parameters
        ----------
        burst : bool, optional
            true = burst mode, false = continuous mode.
        
        Returns
        -------
        burst_mode : bool or None
            only returns if `burst` is None (query mode).
        """

        if burst is None:
            return self.query("TBAS:MODE?").strip() == 'BURS'
        
        self.write(f"TBAS:MODE {'BURS' if burst else 'CONT'}")
        self.write(f"Entered {'burst' if burst else 'continuous mode.'}")
    
    def run(self, on: bool = None) -> bool | None:
        """
        Enable/Disable sequencer or query sequencer state.
        
        Parameters
        ----------
        on : bool, optional
            true = sequencer on, false = sequencer off.
        
        Returns
        -------
        sequencer_state : bool or None
            only returns if `on` is None (query mode).
        """

        if on is None:
            return int(self.query("TBAS:RUN?"))
        
        self.write(f"TBAS:RUN {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled sequencer.")

    def output_enable(self, on: bool):
        """
        Enable or disable all outputs.
        
        Parameters
        ----------
        on : bool
            true = enable, false = disable.
        """

        self.write(f"OUTPut:STATe:ALL {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled outputs.")
        
    def frequency(self, f: float = None) -> float:
        """
        Set or query the internal clock frequency.
        
        Parameters
        ----------
        f : float, optional
            frequency in Hz.
        
        Returns
        -------
        frequency : float
            frequency in Hz.
        """

        if f is None:
            f = float(self.query("TBAS:FREQ?"))
            self._frequency = f
            return f
        
        f = max(5.0e4, min(1.675e9, f))
        self.write(f"TBAS:FREQ {f}")
        self._frequency = f
        self.info(f"Set frequency to {f} Hz.")
        return f
    
    def period(self, T: float = None) -> float:
        """
        Set or query the internal clock period.
        
        Parameters
        ----------
        T : float, optional
            period in s.
        
        Returns
        -------
        period : float
            period in s.
        """

        if T is None:
            T = float(self.query("TBAS:PERiod?"))
            self._frequency = 1 / T
            return T
        
        T = max(1 / 1.675e9, min(1 / 5.0e4, T))
        self.write(f"TBAS:PERiod {T}")
        self._frequency = 1 / T
        self.info(f"Set period to {T} s.")
        return T

    def time_offset(self, d: float = None) -> float:
        """
        Set or query the global time offset post-trigger.
        
        Parameters
        ----------
        d : float, optional
            offset in s.
        
        Returns
        -------
        delay : float
            offset in s.
        """

        if d is None:
            self._time_offset = float(self.query("TBAS:DOFFset?"))
            return self._time_offset
        
        f = self._frequency
        if f is None:
            f = self.frequency()

        d = min(1.0 / f, max(0.0, d))
        self._time_offset = d
        self.write(f"TBAS:OFFset {d}")
        self.info(f"Set global time offset to {d} s.")
        return d
    
    def trigger_input_Z(self, Z: float = None) -> float:
        """
        Set or query the trigger input impedence.
        
        Parameters
        ----------
        Z : float, optional
            input impedence in Ohms (rounded to 1 kOhm or 50 Ohm).
        
        Returns
        -------
        Z : float
            input impedence in Ohms.
        """

        if Z is None:
            return float(self.query("TBAS:TIN:IMPedance?"))
        
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50

        self.write(f"TBAS:TIN:IMPedance {Z}")
        self.info(f"Set trigger input impedence to {Z} Ohm.")

    def trigger_input_level(self, V: float = None) -> float:
        """
        Set or query the trigger input level.
        
        Parameters
        ----------
        V : float, optional
            input level for external trigger in volts.

        Returns
        -------
        V : float
            input level for external trigger in volts.
        """

        if V is None:
            return float(self.query("TBAS:TIN:LEVel?"))
        
        V = max(-5.0, min(5.0, V))
        self.write(f"TBAS:TIN:LEVel {V}")
        self.info(f"Set trigger input level to {V} V.")
        return V

    def trigger_input_slope(self, pos: bool = None) -> bool | None:
        """
        Set or query the trigger input slope.
        
        Parameters
        ----------
        pos : bool
            true = positive slope, false = negative slope.

        Returns
        -------
        positive : bool
            only returns if `pos` is None (query mode).
        """
        
        if pos is None:
            return self.query(f"TBAS:TIN:SLOPe?").strip() == 'POS'
        
        self.write(f"TBAS:TIN:POLarity {'POS' if pos else 'NEG'}")
        self.info(f"Set trigger input polarity {'posi' if pos else 'nega'}tive.")

    def manual_event(self):
        """Manually trigger the DTG."""
        
        self.write(" TBAS:EIN:IMMediate")
        self.info("Manually flagged event on device.")
    
    def event_input_Z(self, Z: float = None) -> float:
        """
        Set or query the event input impedence.
        
        Parameters
        ----------
        Z : float, optional
            input impedence in Ohms (rounded to 1 kOhm or 50 Ohm).
        
        Returns
        -------
        Z : float
            input impedence in Ohms.
        """

        if Z is None:
            return float(self.query("TBAS:EIN:IMPedance?"))
        
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50

        self.write(f"TBAS:EIN:IMPedance {Z}")
        self.info(f"Set event input impedence to {Z} Ohm.")

    def event_input_level(self, V: float = None) -> float:
        """
        Set or query the event trigger input level.
        
        Parameters
        ----------
        V : float, optional
            input level for external event trigger in volts.

        Returns
        -------
        V : float
            input level for external event trigger in volts.
        """

        if V is None:
            return float(self.query("TBAS:EIN:LEVel?"))
        
        V = max(-5.0, min(5.0, V))
        self.write(f"TBAS:EIN:LEVel {V}")
        self.info(f"Set event input level to {V} V.")
        return V

    def event_input_polarity(self, pos: bool = None) -> bool | None:
        """
        Set or query the event trigger input polarity.

        Parameters
        ----------
        pos : bool
            true = positive, false = negative.

        Returns
        -------
        positive : bool
            only returns if `pos` is None (query mode).      
        """
        
        if pos is None:
            return self.query(f"TBAS:EIN:POLarity?").strip() == 'NORM'
        
        self.write(f"TBAS:EIN:POLarity {'NORM' if pos else 'INV'}")
        self.info(f"Set event input polarity {'posi' if pos else 'nega'}tive.")

    def burst_count(self, N: int = None) -> int:
        """
        Set or query the burst count.
        
        Parameters
        ----------
        N : int, optional
            burst count.

        Returns
        -------
        burst_count : int
        """
        
        if N is None:
            self._burst_count = int(self.query("TBAS:COUNt?"))
            return self._burst_count
        
        N = max(1, min(65536, N))
        self._burst_count = N
        self.write(f"TBAS:COUNt {N}")
        self.info(f"Set burst count to {N}.")

    # TODO NEED TO MODIFY THIS BECAUSE OF WEIRDNESS DUE TO LHOLD AND THOLD TYPE STUFF
    # ======================= PHYSICAL CHANNEL SETTINGS =======================

    # --------------------- Pulse Generator Mode Specific ----------------------

    @_mode_required("PULS")
    def prate(self, ch: str | Channel, prate: float = None) -> float | None:
        """
        Set or query the relative rate of a channel, rounded appropriately
        
        Parameters
        ----------
        ch : str or Channel
            target channel.
        prate : float, optional
            rate of the channel relative to the internal clock frequency
            (rounded to 1, 1/2, 1/4, 1/8, 1/16, or OFF).
        
        Returns
        -------
        prate : float
        """

        return self.get_channel(ch).prate(prate)

    @_mode_required("PULS")
    def polarity(self, ch: str | Channel, pos: bool = None) -> bool | None:
        """
        Set or query the polarity of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        pos : bool, optional
            true = positive, false = negative.
        
        Returns
        -------
        polarity : bool or None
            only returns if `pos` is None.
        """

        return self.get_channel(ch).polarity(pos)
    
    @_mode_required("PULS")
    def lead_hold(self, ch: str | Channel, mode: str | None = None) -> str | None:
        """
        Set or query the lead hold parameter of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        mode : str, optional
            One of `LDEL` or `PHAS`.

        Returns
        -------
        lhold : str or None
        """
        return self.get_channel(ch).lhold(mode)
    
    @_mode_required("PULS")
    def trail_hold(self, ch: str | Channel, mode: str | None = None) -> str | None:
        """
        Set or query the trail hold parameter of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        mode : str, optional
            One of `TDEL`, `DCYC`, or `WIDT`.

        Returns
        -------
        thold : str or None
        """
        return self.get_channel(ch).thold(mode)

    @_mode_required("PULS")
    def pulse_width(self, ch: str | Channel, W: float = None) -> float | None:
        """
        Set or query the pulse width of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        W : float, optional
            pulse width in s.
        
        Returns
        -------
        pulse width : float
        """

        return self.get_channel(ch).width(W)

    @_mode_required("PULS")
    def lead_delay(self, ch: str | Channel, l: float = None) -> float | None:
        """
        Set or query the lead delay of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        l : float, optional
            lead delay in s.
        
        Returns
        -------
        lead_delay : float
        """

        return self.get_channel(ch).ldelay(l)
    
    @_mode_required("PULS")
    def trail_delay(self, ch: str | Channel, t: float = None) -> float | None:
        """
        Set or query the trail delay of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        t : float, optional
            trail delay in s.
        
        Returns
        -------
        trail_delay : float
        """

        return self.get_channel(ch).tdelay(t)

    # --------------------------------------------------------------------------

    def low_level(self, ch: str | Channel, V: float = None) -> float | None:
        """
        Set or query the logical low level of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel
        V : float, optional
            logical low level in volts.
        
        Returns
        -------
        logical_low : float
        """

        return self.get_channel(ch).low(V)

    def high_level(self, ch: str | Channel, V: float = None) -> float | None:
        """
        Set or query the logical high level of a channel.

        Parameters
        ----------
        ch : str or Channel
            target channel.
        V : float, optional
            logical high level in volts.

        Returns
        -------
        logical_high : float
        """

        return self.get_channel(ch).high(V)
    
    def chan_output(self, ch: str | Channel, on: bool = None) -> bool | None:
        """
        Set or query the output state of the channel.
        
        Parameters
        ----------
        ch : str or Channel
            target channel.
        on : bool, optional
            true = enabled, false = disabled.
        
        Returns
        -------
        output_state : bool or None
            only returns if `on` is None (query mode). 
        """
        
        return self.get_channel(ch).enable(on)
        
    def termination_Z(self, ch, Z: float = None) -> float | None:
        """
        Set or query the termination impedence of the channel.
        
        Parameters
        ----------
        ch : str or Channel
            target channel.
        Z : float, optional
            impedence in Ohms (rounded to 1 kOhm or 50 Ohms).
        
        Returns
        -------
        Z : float
        """

        return self.get_channel(ch).termination_Z(Z)

    def termination_V(self, ch, V: float = None) -> float:
        """
        Set or query the termination voltage of the channel.
        
        Parameters
        ----------
        ch : str or Channel
            target channel.
        V : float, optional
            termination voltage.
        
        Returns
        -------
        V : float
        """

        return self.get_channel(ch).termination_V(V)

    # ========================== Data Generator Mode ==========================

    # ---------------------------- Handling Groups ----------------------------

    @_mode_required('DATA')
    def new_group(self, name: str, channels: int | List[None | str | Channel]):
        """
        Define a new group.
        
        Parameters
        ----------
        name : str
            name of the new group
        channels : int or list of str, optional
            width of the new group or list of channels to be included;
            empty slots should be represented by None.
        """

        if type(channels) == int:
            N = channels
        else:
            N = len(channels)

        self.groups[name] = Group(name, N)
        self.write(f'GROup:NEW "{name}", {N}')
        self.info(f"Created new group {name} of width {N}")

        if type(channels) != int:
            self.assign_signals(name, channels)

    @_mode_required('DATA')
    def assign_signals(self, group_name: str, 
                       channels: List[None | str | Channel]):
        """
        Assign physical channels to a group.
        
        Parameters
        ----------
        group_name : str
        channels : list of str, optional
            list of channels to be included; empty slots should be 
            represented by None.
        """

        width = self.groups[group_name].width
        if len(channels) != width:
            self.error(
                f"Channels list of size {len(channels)} "
                f"does not match the group width {width}"
            )
        
        for i, ch in enumerate(channels):
            if ch is None: 
                continue
            self.assign_signal(group_name, i, ch)

    @_mode_required('DATA')
    def assign_signal(self, group_name: str, idx: int, ch: str | Channel):
        """
        Assign a physical channel to a group slot.
        
        Parameters
        ----------
        group_name : str
        idx : int
            index at which to assign the channel.
        ch : str or Channel
            channel to assign.
        """

        if idx >= self.groups[group_name].width:
            self.error(f"Index {idx} is out of bounds for group {group_name}")
            return

        ch = self.get_channel(ch)
        if ch is None:
            self.error(f"Unrecognized channel name {ch}")
            return
        
        self.groups[group_name].channels[idx] = ch
        self.write(f'SIGNal:ASSign "{group_name}[{idx}]", "{ch._id()}"')
        self.info(
            f"Added channel {ch._id()} to group element {group_name}[{idx}]"
        )

    @_mode_required('DATA')
    def delete_group(self, name: str):
        """
        Delete a group.

        Parameters
        ----------
        name : str
            group name.
        """

        del self.groups[name]
        self.write(f'GROup:DELete "{name}"')
        self.info(f"Deleted group: {name}")

    @_mode_required('DATA')
    def clear_groups(self):
        """Delete all groups."""

        self.groups.clear()
        self.write("GROup:DELete:ALL")
        self.info("Deleted all groups")

    # ---------------------------- Handling Blocks ----------------------------    
    @_mode_required('DATA')
    def new_block(self, name: str, N: int):
        """
        Create a new block of a given length
        
        Parameters
        ----------
        name : str
            name of new block.
        N : int
            length of the block.
        """

        self.blocks[name] = Block(name, N)
        self.write(f'BLOCk:NEW "{name}", {N}')
        self.info(f"Created new block {name} of length {N}")

    @_mode_required('DATA')
    def add_block_data(self, block_name:str, group_name: str, ch_idx: int,
                    data: str | bytes, start_idx: int = 0, num_bits: int = None):
        """
        Add data to the block associated to some logical channel.
        
        Parameters
        ----------
        block_name : str
        group_name : str
            group to associate with this section of the block.
        ch_idx : int
            particular channel index within the group to associate.
        data : str or bytes
            data to insert into the block eg. ("10..." or b'\\x01...').
        start_idx : int, default=0
            index within the block at which to start the insertion.
        num_bits : int, default=None
            number of data bits to insert (by default, all of them).
        """

        B = self.blocks[block_name]
        G = self.groups[group_name]

        data = bitarray(data)
        if num_bits is None:
            num_bits = len(data)
        
        B.add_data(G, ch_idx, data, offset = start_idx, length = num_bits)
        self.write(f'BLOCk:SELect "{block_name}"')
        payload = data.tobytes()
        header = f'#{len(str(len(payload)))}{len(payload)}'.encode('ascii')
        cmd = f'SIGNal:BDATa "{group_name}[{ch_idx}]", {start_idx}, {num_bits}, '
        cmd = cmd.encode('ascii')
        cmd += header + payload
        self.write_raw(cmd)

        self.info(
            f"Loaded block {block_name} with {header}{payload} from index "
            f"{start_idx} to {start_idx + num_bits} associated to group "
            f"{group_name} channel {ch_idx}"
        )

    @_mode_required('DATA')
    def delete_block(self, name: str):
        """
        Delete a block.
        
        Parameters
        ----------
        name : str
            name of the block to delete.
        """

        del self.blocks[name]
        self.write(f'BLOCk:DELete "{name}"')
        self.info(f"Deleted block {name}")

    @_mode_required('DATA')
    def clear_blocks(self):
        """Delete all blocks."""

        self.blocks.clear()
        self.write("BLOCk:DELete:ALL")
        self.info("Deleted all blocks")

    # --------------------------- Handling Sequences ---------------------------

    @_mode_required('DATA')
    def sequence_length(self, N: int = None) -> int:
        """
        Set or query the sequence length. The sequence essentially functions
        like a low level script. This determines the number of lines.
        
        Parameters
        ----------
        N : int or None
            sequence length.

        Returns
        -------
        sequence_length : int
        """

        if N is None:
            self.seq_length = int(self.query("SEQuence:LENGth?"))
            return  self.seq_length

        N = max(0, min(8000, N))
        self.sequence = [""]*N         
        self.seq_length = N
        self.write("SEQuence:LENGth?")
        self.info(f"Set sequence length to {N}")
        return N
    
    @_mode_required('DATA')
    def get_sequence_line(self, idx: int) -> str:
        """
        Query a line of the sequence.
        
        Parameters
        ----------
        idx : int
            index of the line to query.

        Returns
        -------
        line : str
            string representation of the sequence line.
        """

        self.sequence[idx] = self.query(f"SEQuence:DATA? {idx}").strip()
        return self.sequence[idx]
    
    @_mode_required('DATA')
    def set_sequence_line(self, idx: int, label: str, wait_trigger: bool,
                block_name: str, repetitions: int, jump_to: str, go_to: str):
        """
        Set a line of the sequence. Consult the DTG manual for a more thorough
        explanation of these.
        
        Parameters
        ----------
        idx : int
            index of the line within the sequence.
        label : str
            label used to jump to this line. 'START' is used for the
            start of a program; often this can be kept blank ('').
        wait_trigger : bool
            whether to await a trigger signal after executing.
        block_name : str
            name of the block to play (each line emits data encoded by
            some block on the associated logical channels).
        repetitions : int
            number of times to repeat the block.
        jump_to : str
            label to jump to on a trigger; often left blank ('').
        go_to : str
            label to jump to after execution; often left blank (''),
            meaning: continue to the next line.
        """

        line  = f'"{label}", {int(wait_trigger)}, "{block_name}", '
        line += f'{repetitions}, "{jump_to}", "{go_to}"'
        self.sequence[idx] = line
        self.write(f"SEQ:DATA {idx}, {line}")
        self.info(f"Assigned line {idx} in sequence: {line}")

    # =========================== Simulating Outputs ===========================

    @_mode_required('PULS')
    def simulate_puls_output(self):
        """
        Plot a simulation of the PULS mode output given current settings.
        
        Returns
        -------
        fig : Figure
        axes : list of Axes
        """

        fig, axes = plt.subplots(len(self.channels), 1, sharex = True)
        period = self.period()
        for ch, ax in zip(self.channels.keys(), axes):
            chan = self.channels[ch]

            on = self.chan_output(chan)
            if not on:
                ax.plot([0.0, 16 * period], [0.0, 0.0], color = 'r')
                ax.set_ylabel(chan.name)
                continue
            
            prate = self.prate(chan)
            ldelay = self.lead_delay(chan)
            tdelay = self.trail_delay(chan)
            polarity = self.polarity(chan)
            low = self.low_level(ch)
            high = self.high_level(ch)

            dt = chan.rise_time
            ch_period = period / prate

            if polarity:
                logic_l = low
                logic_h = high
            else:
                logic_h = low
                logic_l = high

            t_chunk = np.array([0.0, ldelay, ldelay + dt, tdelay, tdelay + dt])
            v_chunk = np.array([logic_l, logic_l, logic_h, logic_h, logic_l])
            nreps = int(16 * prate)
            T = np.zeros(5 * nreps + 1)
            V = np.zeros(5 * nreps + 1)
            for i in range(nreps):
                T[5 * i: 5 * (i + 1)] += t_chunk + i * ch_period
                V[5 * i: 5 * (i + 1)] += v_chunk
            T[-1] = 16 * period
            V[-1] = logic_l

            ax.plot(T, V, color = 'r')
            ax.set_ylabel(chan.name)
        
        ax.set_xlabel("s")
        return fig, axes

    @_mode_required('DATA')
    def simulate_block_output(self, block_name: str):
        """
        Plot a simulation of the output of some specified block
        
        Parameters
        ----------
        block_name : str
            name of the block to simulate.

        Returns
        -------
        fig : Figure
        Axes : list of Axes
        """

        period = self.period()
        B = self.blocks[block_name]
        dtg_channels = list(self.channels.keys())
        fig, axes = plt.subplots(len(dtg_channels), 1, sharex = True)

        for wf in B.waveforms:
            ch = wf.group.channels[wf.ch_idx]
            if ch is None:
                continue

            toff = period * wf.off
            # low = ch.low()
            # high = ch.high()
            low = 0
            high = 1
            dt = ch.rise_time

            S = low + (high - low) * wf.to_array()
            T = [toff]
            V = [S[0]]

            curr_v = np.nan
            for j, v in enumerate(S):
                if v != curr_v:
                    T.extend([toff + j * period, toff + j * period + dt])
                    V.extend([curr_v, v])
                    curr_v = v

            T.append(toff + period * len(S))
            V.append(S[-1])

            ax_ind = dtg_channels.index(ch.name)
            axes[ax_ind].plot(T, V, color = 'r')
            axes[ax_ind].set_ylabel(ch.name)

        axes[-1].set_xlabel("s")
        return fig, axes
    
    # --------------------------- Save / Load State ---------------------------

    def save_state_dev(self, path: str):
        """
        Save the DTG state on the device. Use '.dtg' or '.dat' extension.
        
        Parameters
        ----------
        path : str
            directory at which to save.
        """
        self.write(f'MMEMory:STORe "{path}"')
        self.info(f"Saved DTG state to device at {path}")

    def load_state_dev(self, path: str):
        """
        Load the DTG state from the device. Use '.dtg' or '.dat' extension.
        
        Parameters
        ----------
        path : str
            directory from which to load.
        """
        self.write(f'MMEMory:LOAD "{path}"')
        self.info(f"Loaded DTG state from device at {path}")

    def save_state_json(self, path: str):
        """
        Save the DTG state to JSON locally.

        Parameters
        ----------
        path : str
            directory at which to save.
        """
        super().save_state_json(path)

    def load_state_json(self, path: str):
        """
        Load the DTG state from JSON locally.
        
        Parameters
        ----------
        path : str
            directory from which to load.
        """
        super().load_state_json(path)

    def _serialize_state(self) -> dict:
        state = {
            'version'       : 1,
            'mode'          : self.mode,
            'frequency'     : self._frequency,
            'time_offset'   : self._time_offset,
            'burst_count'   : self._burst_count,
            'channels': {
                name: {
                    'enabled'   : ch.enable(),
                    'polarity'  : ch.polarity(),
                    'high'      : ch.high(),
                    'low'       : ch.low(),
                    'width'     : ch.width(),
                    'lhold'     : ch.lhold(),
                    'thold'     : ch.thold(),
                    'ldelay'    : ch.ldelay(),
                    'tdelay'    : ch.tdelay(),
                    'prate'     : ch.prate(),
                    'termination_Z': ch.termination_Z(),
                    'termination_V': ch.termination_V(),
                }
                for name, ch in self.channels.items()
            }
        }

        if self.mode == 'DATA':
            # Serialize groups
            state['groups'] = {
                name: {
                    'width'     : group.width,
                    'channels'  : [ch._id() if ch else None 
                                   for ch in group.channels]
                }
                for name, group in self.groups.items()
            }

            # Serialize blocks
            state['blocks'] = {}
            for name, block in self.blocks.items():
                bdict = {'length': block.length, 'waveforms': []}
                for wf in block.waveforms:
                    # Encode binary data efficiently
                    raw_bytes = wf.bits.tobytes()
                    encoded = base64.b64encode(raw_bytes).decode('ascii')
                    bdict['waveforms'].append({
                        'group' : wf.group.name,
                        'ch_idx': wf.ch_idx,
                        'offset': wf.off,
                        'bitlen': len(wf.bits),
                        'data_b64': encoded
                    })
                state['blocks'][name] = bdict

            # Serialize sequencer
            state['sequence'] = [
                {
                    'index' : i,
                    'line'  : line
                }
                for i, line in enumerate(self.sequence)
                if line
            ]

        return state

    def _deserialize_state(self, state: dict):
        self.operation_mode(state['mode'] == 'PULS')
        self.frequency(state['frequency'])
        self.time_offset(state['time_offset'])
        self.burst_count(state['burst_count'])

        for ch_id, ch_state in state['channels'].items():
            if ch_id not in self.channels:
                continue

            ch = self.channels[ch_id]
            self.chan_output(ch, ch_state['enabled'])
            self.low_level(ch, ch_state['low'])
            self.high_level(ch, ch_state['high'])
            self.termination_Z(ch, ch_state['termination_Z'])
            self.termination_V(ch, ch_state['termination_V'])
            if self.mode == 'PULS':
                self.polarity(ch, ch_state['polarity'])
                self.prate(ch, ch_state['prate'])
                self.lead_hold(ch, ch_state['lhold'])
                self.trail_hold(ch, ch_state['thold'])
                self.lead_delay(ch, ch_state['ldelay'])
                self.trail_delay(ch, ch_state['tdelay'])
                self.pulse_width(ch, ch_state['width'])

        if self.mode == 'DATA':
            self.clear_groups()
            self.clear_blocks()

            for name, group in state['groups'].items():
                self.new_group(name, group['width'])
                self.assign_signals(name, group['channels'])

            for name, bdict in state['blocks'].items():
                self.new_block(name, bdict['length'])
                for wf in bdict['waveforms']:
                    raw = base64.b64decode(wf['data_b64'])
                    bits = bitarray()
                    bits.frombytes(raw)
                    bits = bits[:wf['bitlen']]
                    self.add_block_data(
                        name, wf['group'], wf['ch_idx'],
                        bits, wf['offset'], wf['bitlen']
                    )

            indices = [l['index'] for l in state['sequence']]
            if indices:
                seq_len = max(indices) + 1
            else:
                return

            self.sequence_length(seq_len)
            for entry in state['sequence']:
                idx = entry['index']
                fields = eval(entry['line']) # assumes safe inputs
                self.set_sequence_line(idx, *fields)

"""DTG5274 instrument class"""

class dtg5274(DTG):
    """Class interface for controlling the DTG5274"""
    def __init__(self, logger = None, instrument_id: str = None):
        """
        Parameters
        ----------
        logger : Logger, optional
            logger used by abstractDevice.
        instrument_id : str, optional
            VISA resource name.
        """

        self.pyvisa_config = {
            "resource_name"     : "GPIB0::27::INSTR",
            "output_buffer_size": 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
            'max_retries': 3,
            'retry_delay': 0.1,
            'min_interval': 0.05
        }

        self.MODULES = {
            2: {'n_ch': 2, 'min_V': -1.0, 'max_V': 2.5, 'min_V_diff': 0.1, 'min_width': 2.9e-10, 'rise_time': 340e-12}, # DTGM20
            4: {'n_ch': 2, 'min_V': -1.0, 'max_V': 2.7, 'min_V_diff': 0.1, 'min_width': 2.9e-10, 'rise_time': 350e-12}, # DTGM21
        }
        self.mainframes = [1,]
        self.slots      = ['A', 'B', 'C', 'D']

        super().__init__(logger, instrument_id)
