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
from typing import Callable, Dict, List, Optional

"""Base DTG class"""

class DTG(pyvisaDevice):
    def __init__(self, logger = None, instrument_id: str = None,
                 pyvisa_config = None, MODULES = None, 
                 mainframes = None, slots = None):
        
        if self.pyvisa_config is None:
            self.pyvisa_config = pyvisa_config
        if self.MODULES is None:
            self.MODULES = MODULES
        if self.mainframes is None:
            self.mainframes = mainframes
        if self.slots is None:
            self.slots = slots

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

        self.get_installed_modules()
        self.operation_mode()
    
    def mode_required(*allowed_modes) -> Callable:
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

    def get_installed_modules(self) -> dict:
        """Determine installed hardware modules"""

        for mf in self.mainframes:
            for slot in self.slots:
                module_id = int(self.device.query(f"PGEN{slot}{mf}:ID?"))
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
                            rise_time   = self.MODULES[module_id]['rise_time']
                        )

        return self.modules
    
    def get_channel(self, ch: str | Channel) -> Optional[Channel]:
        if isinstance(ch, Channel):
            return ch
        
        if not ch in self.channels:
            self.error(f"Channel '{ch}' does not exist.")
            return
        
        return self.channels[ch]
    
    def operation_mode(self, pg: bool = None) -> Optional[bool]:
        """Set or query the operation mode."""

        if pg is None:
            self.mode = self.device.query("TBAS:OMODE?").strip()
            return self.mode == 'PULS'
        
        self.mode = 'PULS' if pg else 'DATA'
        self.device.write(f"TBAS:OMODE {self.mode}")
        self.info(f"Set operational mode to {'pulse' if pg else 'data'} mode.")

    def burst_mode(self, burst: bool = None) -> Optional[bool]:
        """
        Set or query whether we are in burst or continuous mode
        True = burst
        """

        if burst is None:
            return self.device.query("TBAS:MODE?").strip() == 'BURS'
        
        self.device.write(f"TBAS:MODE {'BURS' if burst else 'CONT'}")
        self.device.write(f"Entered {'burst' if burst else 'continuous mode.'}")
    
    def run(self, on: bool = None) -> Optional[bool]:
        """Enable/Disable sequencer or query sequencer state."""

        if on is None:
            return int(self.device.query("TBAS:RUN?"))
        
        self.device.write(f"TBAS:RUN {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled sequencer.")

    def output_enable(self, on: bool):
        """Enable or disable all outputs"""

        self.device.write(f"OUTPut:STATe:ALL {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled outputs.")
        
    def frequency(self, f: float = None) -> float:
        """Set or query the internal clock frequency."""

        if f is None:
            f = float(self.device.query("TBAS:FREQ?"))
            self._frequency = f
            return f
        
        f = max(5.0e4, min(1.675e9, f))
        self.device.write(f"TBAS:FREQ {f}")
        self._frequency = f
        self.info(f"Set frequency to {f} Hz.")
        return f
    
    def period(self, T: float = None) -> float:
        """Set or query the internal clock period."""

        if T is None:
            T = float(self.device.query("TBAS:PERiod?"))
            self._frequency = 1 / T
            return T
        
        T = max(1 / 1.675e9, min(1 / 5.0e4, T))
        self.device.write(f"TBAS:PERiod {T}")
        self._frequency = 1 / T
        self.info(f"Set period to {T} s.")
        return T

    def time_offset(self, d: float = None) -> float:
        """Set or query the global time offset post-trigger"""

        if d is None:
            self._time_offset = float(self.device.query("TBAS:DOFFset?"))
            return self._time_offset
        
        f = self._frequency
        if f is None:
            f = self.frequency()

        d = min(1.0 / f, max(0.0, d))
        self._time_offset = d
        self.device.write(f"TBAS:OFFset {d}")
        self.info(f"Set global time offset to {d} s.")
        return d
    
    def trigger_input_Z(self, Z: float = None) -> float:
        """Set or query the trigger input impedence"""

        if Z is None:
            return float(self.device.query("TBAS:TIN:IMPedance?"))
        
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50

        self.device.write(f"TBAS:TIN:IMPedance {Z}")
        self.device.info(f"Set trigger input impedence to {Z} Ohm.")

    def trigger_input_level(self, V: float = None) -> float:
        """Set or query the trigger input level"""

        if V is None:
            return float(self.device.query("TBAS:TIN:LEVel?"))
        
        V = max(-5.0, min(5.0, V))
        self.device.write(f"TBAS:TIN:LEVel {V}")
        self.device.info(f"Set trigger input level to {V} V.")
        return V

    def trigger_input_slope(self, pos: bool = None) -> Optional[bool]:
        """Set or query the trigger input slope"""
        
        if pos is None:
            return self.device.query(f"TBAS:TIN:SLOPe?").strip() == 'POS'
        
        self.device.write(f"TBAS:TIN:POLarity {'POS' if pos else 'NEG'}")
        self.info(f"Set trigger input polarity {'posi' if pos else 'nega'}tive.")

    def manual_event(self):
        """Manually trigger the DTG."""
        
        self.device.write(" TBAS:EIN:IMMediate")
        self.info("Manually flagged event on device.")
    
    def event_input_Z(self, Z: float = None) -> float:
        """Set or query the event input impedence"""

        if Z is None:
            return float(self.device.query("TBAS:EIN:IMPedance?"))
        
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50

        self.device.write(f"TBAS:EIN:IMPedance {Z}")
        self.device.info(f"Set event input impedence to {Z} Ohm.")

    def event_input_level(self, V: float = None) -> float:
        """Set or query the event input level"""

        if V is None:
            return float(self.device.query("TBAS:EIN:LEVel?"))
        
        V = max(-5.0, min(5.0, V))
        self.device.write(f"TBAS:EIN:LEVel {V}")
        self.device.info(f"Set event input level to {V} V.")
        return V

    def event_input_polarity(self, pos: bool = None) -> Optional[bool]:
        """Set or query the event input polarity"""
        
        if pos is None:
            return self.device.query(f"TBAS:EIN:POLarity?").strip() == 'NORM'
        
        self.device.write(f"TBAS:EIN:POLarity {'NORM' if pos else 'INV'}")
        self.info(f"Set event input polarity {'posi' if pos else 'nega'}tive.")

    def burst_count(self, N: int = None) -> int:
        """Set or query the burst count"""
        
        if N is None:
            self._burst_count = int(self.device.query("TBAS:COUNt?"))
            return self._burst_count
        
        N = max(1, min(65536, N))
        self._burst_count = N
        self.device.write(f"TBAS:COUNt {N}")
        self.info(f"Set burst count to {N}.")

    # ======================= PHYSICAL CHANNEL SETTINGS =======================

    # --------------------- Pulse Generator Mode Specific ----------------------

    @mode_required("PULS")
    def prate(self, ch: str | Channel, prate: float = None) -> float:
        """Set or query the relative rate of channel, rounded appropriately"""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        relrates = {
            'NORM': 1.0, 
            'HALF': 0.5, 
            'QUAR': 0.25, 
            'EIGH': 0.125, 
            'SIXT': 0.0625, 
            'OFF': 0
        }

        if prate is None:
            ch.prate = relrates[self.device.query(f"{ch}:PRATE?").strip()]
            return ch.prate
        
        if prate >= 1.0:
            val = 'NORM'
        elif prate >= 0.5:
            val = 'HALF'
        elif prate >= 0.25:
            val = 'QUAR'
        elif prate >= 0.125:
            val = 'EIGH'
        elif prate >= 0.0625:
            val = 'SIXT'
        else:
            val = 'OFF'

        ch.prate = relrates[val]
        self.device.write(f"{ch}:PRATE {val}")
        self.info(f"Set relative rate of channel {ch._id()} to {ch.prate}.")

    @mode_required("PULS")
    def polarity(self, ch: str | Channel, pos: bool = None) -> Optional[bool]:
        """Set or query the polarity of channel."""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if pos is None:
            ch.polarity = self.device.query(f"{ch}:POLarity?") == 'NORM\n'
            return ch.polarity
        
        ch.polarity = pos
        self.device.write(f"{ch}:POLarity {'NORM' if pos else 'INV'}")
        self.info(
            f"Set polarity of channel {ch._id()} to "
            f"{'posi' if pos else 'nega'}tive."
        )

    @mode_required("PULS")
    def pulse_width(self, ch: str | Channel, W: float = None) -> float:
        """Set or query the pulse width of a channel."""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if W is None:
            ch.width = float(self.device.query(f"{ch}:WIDTh?"))
            return ch.width

        f = self._frequency
        if f is None:
            f = self.frequency()
        prate = ch.prate
        if prate is None:
            prate = self.prate(ch)

        W = min(1.0 / (f * prate), max(ch.min_width, W))
        ch.width = W
        self.device.write(f"{ch}:WIDTh {W}")
        self.info(f"Set pulse width of channel {ch._id()} to {W} s.")
        return W

    @mode_required("PULS")
    def lead_delay(self, ch: str | Channel, l: float = None) -> float:
        """Set or query the lead delay of a channel"""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if l is None:
            ch.ldelay = float(self.device.query(f"{ch}:LDELay?"))
            return ch.ldelay
        
        f = self._frequency
        if f is None:
            f = self.frequency()

        l = min(1.0 / f, max(0.0, l))
        ch.ldelay = l
        self.device.write(f"{ch}:LDELay {l}")
        self.info(f"Set lead delay on channel {ch._id()} to {l} s.")
        return l
    
    @mode_required("PULS")
    def trail_delay(self, ch: str | Channel, t: float = None) -> float:
        """Set or query the trail delay of a channel"""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if t is None:
            ch.tdelay = float(self.device.query(f"{ch}:TDELay?"))
            return ch.tdelay
        
        f = self._frequency
        if f is None:
            f = self.frequency()
        prate = ch.prate
        if prate is None:
            prate = self.prate(ch)
        ldelay = ch.ldelay
        if ldelay is None:
            ldelay = self.lead_delay(ch)

        t = min(ldelay + 1.0 / (f * prate), max(ldelay + ch.min_width, t))
        ch.ldelay = t
        self.device.write(f"{ch}:TDELay {t}")
        self.info(f"Set tral delay on channel {ch._id()} to {t} s.")
        return t

    # --------------------------------------------------------------------------

    def low_level(self, ch: str | Channel, V: float = None, force: bool = False
                  ) -> float:
        """Set or query the low level for a channel."""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if V is None:
            ch.low = float(self.device.query(f"{ch}:LOW?"))
            return ch.low
        
        if force:
            h = ch.max_V - ch.min_V_diff
        else:
            high = ch.high 
            if ch.high is None:
                high = self.high_level(ch)
            h = high - ch.min_V_diff

        V = min(h, max(ch.min_V, V))
        ch.low = V
        self.device.write(f"{ch}:LOW {V}")
        self.info(f"Set logical low level of channel {ch._id()} to {V} V.")
        return V

    def high_level(self, ch: str | Channel, V: float = None, force: bool = False
                   ) -> float:
        """Set or query the high level for a channel."""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if V is None:
            ch.high = float(self.device.query(f"{ch}:HIGH?"))
            return ch.high
        
        if force:
            l = ch.min_V + ch.min_V_diff
        else:
            low = ch.low
            if low is None: 
                low = self.low_level(ch)
            l = low + ch.min_V_diff

        V = min(ch.max_V, max(l, V))
        ch.high = V
        self.device.write(f"{ch}:HIGH {V}")
        self.info(f"Set logical low level of channel {ch._id()} to {V} V.")
        return V
    
    def chan_output(self, ch: str | Channel, on: bool = None) -> Optional[bool]:
        """Set or query the output state of the channel."""
        
        ch = self.get_channel(ch)
        if ch is None:
            return

        if on is None:
            ch.enabled = int(self.device.query(f"{ch}:OUTPut?"))
            return ch.enabled
        
        ch.enabled = on
        self.device.write(f"{ch}:OUTPut {'ON' if on else 'OFF'}")
        self.info(f"{'En' if on else 'Dis'}abled channel {ch._id()}.")

    def termination_Z(self, ch, Z: float = None) -> float:
        """Set or query the termination impedence of the channel."""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if Z is None:
            ch.termination_Z = float(self.device.query(f"{ch}:TIMPedance?"))
            return ch.termination_Z
        
        if Z >= 1e3:
            Z = 1e3
        else:
            Z = 50

        ch.termination_Z = Z
        self.device.write(f"{ch}:TIMPedance {Z}")
        self.info(f"Set termination impedance on channel {ch._id()} to {Z} Ohm.")
        return Z

    def termination_V(self, ch, V: float = None) -> float:
        """Set or query the termination voltage of the channel"""

        ch = self.get_channel(ch)
        if ch is None:
            return
        
        if V is None:
            ch.termination_V = float(self.device.query(f"{ch}:TVOLtage?"))
            return ch.termination_V
        
        V = max(-2.0, min(5.0, V))
        ch.termination_V = V
        self.device.write(f"{ch}:TVOLtage {V}")
        self.info(f"Set termination voltage on channel {ch._id()} to {V} V.")
        return V

    # ========================== Data Generator Mode ==========================

    # ---------------------------- Handling Groups ----------------------------

    @mode_required('DATA')
    def new_group(self, name: str, channels: int | List[None | str | Channel]):
        """Define a new group."""

        if type(channels) == int:
            N = channels
        else:
            N = len(channels)

        self.groups[name] = Group(name, N)
        self.device.write(f'GROup:NEW "{name}", {N}')
        self.info(f"Created new group {name} of width {N}")
        self.assign_signals(name, channels)

    @mode_required('DATA')
    def assign_signals(self, group_name: str, 
                       channels: List[None | str | Channel]):
        """Assign physical channels to a group."""

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

    @mode_required('DATA')
    def assign_signal(self, group_name: str, idx: int, ch: str | Channel):
        """Assign a physical channel to a group slot."""

        if idx >= self.groups[group_name].width:
            self.error(f"Index {idx} is out of bounds for group {group_name}")
            return

        ch = self.get_channel(ch)
        if ch is None:
            self.error(f"Unrecognized channel name {ch}")
            return
        
        self.groups[group_name].channels[idx] = ch
        self.device.write(f'SIGNal:ASSign "{group_name}[{idx}]", "{ch._id()}"')
        self.info(
            f"Added channel {ch._id()} to group element {group_name}[{idx}]"
        )

    @mode_required('DATA')
    def delete_group(self, name: str):
        """Delete a group."""

        del self.groups[name]
        self.device.write(f'GROup:DELete "{name}"')
        self.info(f"Deleted group: {name}")

    @mode_required('DATA')
    def clear_groups(self):
        """Delete all groups."""

        self.groups.clear()
        self.device.write("GROup:DELete:ALL")
        self.info("Deleted all groups")

    # ---------------------------- Handling Blocks ----------------------------
    
    
    @mode_required('DATA')
    def new_block(self, name: str, N: int):
        """Create a new block of a given length"""

        self.blocks[name] = Block(name, N)
        self.device.write(f'BLOCk:NEW "{name}", {N}')
        self.info(f"Created new block {name} of length {N}")

    @mode_required('DATA')
    def add_block_data(self, block_name:str, group_name: str, ch_idx: int,
                    data: str | bytes, start_idx: int = 0, num_bits: int = None):
        """Add data to the block associated to some logical channel"""

        B = self.blocks[block_name]
        G = self.groups[group_name]

        data = bitarray(data)
        if num_bits is None:
            num_bits = len(data)
        
        B.add_data(G, ch_idx, data, offset = start_idx, length = num_bits)
        self.device.write(f'BLOCk:SELect "{block_name}"')
        payload = data.tobytes()
        header = f'#{len(str(len(payload)))}{len(payload)}'.encode('ascii')
        cmd = f'SIGNal:BDATa "{group_name}[{ch_idx}]", {start_idx}, {num_bits}, '
        cmd = cmd.encode('ascii')
        cmd += header + payload
        self.device.write_raw(cmd)

        self.info(
            f"Loaded block {block_name} with {header}{payload} from index "
            f"{start_idx} to {start_idx + num_bits} associated to group "
            f"{group_name} channel {ch_idx}"
        )

    @mode_required('DATA')
    def delete_block(self, name: str):
        """Delete a block."""

        del self.blocks[name]
        self.device.write(f'BLOCk:DELete "{name}"')
        self.info(f"Deleted block {name}")

    @mode_required('DATA')
    def clear_blocks(self):
        """Delete all blocks."""

        self.blocks.clear()
        self.device.write("BLOCk:DELete:ALL")
        self.info("Deleted all blocks")

    # --------------------------- Handling Sequences ---------------------------

    @mode_required('DATA')
    def sequence_length(self, N: int = None) -> int:
        """Set or query the sequence length."""

        if N is None:
            self.seq_length = int(self.device.query("SEQuence:LENGth?"))
            return  self.seq_length

        N = max(0, min(8000, N))
        self.sequence = [""]*N         
        self.seq_length = N
        self.device.write("SEQuence:LENGth?")
        self.info(f"Set sequence length to {N}")
        return N
    
    @mode_required('DATA')
    def get_sequence_line(self, idx: int):
        """Query a line of the sequence"""

        self.sequence[idx] = self.device.query(f"SEQuence:DATA? {idx}").strip()
        return self.sequence[idx]
    
    @mode_required('DATA')
    def set_sequence_line(self, idx: int, label: str, wait_trigger: bool,
                block_name: str, repetitions: int, jump_to: str, go_to: str):
        """Set a line of the sequence """

        line  = f'"{label}", {int(wait_trigger)}, "{block_name}", '
        line += f'{repetitions}, "{jump_to}", "{go_to}"'
        self.sequence[idx] = line
        self.device.write(f"SEQ:DATA {idx}, {line}")
        self.info(f"Assigned line {idx} in sequence: {line}")

    # =========================== Simulating Outputs ===========================

    @mode_required('PULS')
    def simulate_puls_output(self):
        """Plot a simulation of the PULS mode output given current settings."""

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

    @mode_required('DATA')
    def simulate_block_output(self, block_name: str):
        """Plot a simulation of the output of some specified block"""

        period = self.period()
        B = self.blocks[block_name]
        dtg_channels = list(self.channels.keys())
        fig, axes = plt.subplots(len(dtg_channels), 1, sharex = True)

        for wf in B.waveforms:
            ch = wf.group.channels[wf.ch_idx]
            if ch is None:
                continue

            toff = period * wf.off
            if ch.low is None:
                low = self.low_level(ch)
            else:
                low = ch.low
            if ch.high is None:
                high = self.high_level(ch)
            else:
                high = ch.high
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
        """Save the DTG state on the device. Use '.dtg' or '.dat' extension"""
        self.device.write(f'MMEMory:STORe "{path}"')
        self.info(f"Saved DTG state to device at {path}")

    def load_state_dev(self, path: str):
        """Load the DTG state from the device. Use '.dtg' or '.dat' extension"""
        self.device.write(f'MMEMory:LOAD "{path}"')
        self.info(f"Loaded DTG state from device at {path}")

    def save_state_json(self, path: str):
        """Save the DTG state to json on computer"""
        state = self._serialize_state()
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state_json(self, path: str):
        """Load the DTG state from json on computer"""
        with open(path, 'r') as f:
            state = json.load(f)
        self._deserialize_state(state)

    def _serialize_state(self) -> dict:
        state = {
            'version'       : 1,
            'mode'          : self.mode,
            'frequency'     : self._frequency,
            'time_offset'   : self._time_offset,
            'burst_count'   : self._burst_count,
            'channels': {
                name: {
                    'enabled'   : ch.enabled,
                    'polarity'  : ch.polarity,
                    'high'      : ch.high,
                    'low'       : ch.low,
                    'width'     : ch.width,
                    'ldelay'    : ch.ldelay,
                    'tdelay'    : ch.tdelay,
                    'prate'     : ch.prate,
                    'termination_Z': ch.termination_Z,
                    'termination_V': ch.termination_V
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
    def __init__(self, logger = None, instrument_id: str = None):
        
        self.pyvisa_config = {
            "resource_name"     : "GPIB0::27::INSTR",
            "output_buffer_size": 512,
            "gpib_eos_mode"     : False,
            "gpib_eos_char"     : ord('\n'),
            "gpib_eoi_mode"     : True,
        }

        self.MODULES = {
            2: {'n_ch': 2, 'min_V': -1.0, 'max_V': 2.5, 'min_V_diff': 0.1, 'min_width': 2.9e-10, 'rise_time': 340e-12}, # DTGM20
            4: {'n_ch': 2, 'min_V': -1.0, 'max_V': 2.7, 'min_V_diff': 0.1, 'min_width': 2.9e-10, 'rise_time': 350e-12}, # DTGM21
        }
        self.mainframes = [1,]
        self.slots      = ['A', 'B', 'C', 'D']

        super().__init__(logger, instrument_id)
