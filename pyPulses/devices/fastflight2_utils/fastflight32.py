from comtypes.client import CreateObject # type: ignore
from comtypes.automation import VARIANT # type: ignore
from typing import Tuple
import time

try:
    # If the library has already been generated, import it
    import comtypes.gen.FF2CTRLLib as FF2Lib # type: ignore
except:
    # Generate the library from the DLL, then import it
    from comtypes.client import GetModule # type: ignore
    GetModule(r'C:\Windows\SysWOW64\FF2Ctrl.dll')
    import comtypes.gen.FF2CTRLLib as FF2Lib # type: ignore

class FastFlight32():
    def __init__(self):
        self._num_spectra_per_trace = 1
        self.dither_len = 0.0           # Dithering length to use (V)
        self.dither_ready = False       # Are our protocols set for dithering?

        self.FF2Ctrl = CreateObject(FF2Lib.FF2CtrlObj)
        self.GSObj   = CreateObject(FF2Lib.GSObj)
        self.TOFObj  = CreateObject(FF2Lib.TOFObj)
        self.Protocols = [CreateObject(FF2Lib.ProtocolObj) 
                          for _ in range(16)]
        
        self.protocol_parms = [
            'CompType',             # 0 - lossy, 1 - lossless, 2 - stick (unused)
            'EnableCNS',            # 1 - enabled, 0 - disabled
            'PrecEnhEnable',        # 1 - enabled, 0 - disabled
            'RecordLength',         # in units of us
            'RecordsPerSpectrum',   # between 1 and 65_535
            'SamplingInterval',     # 0 - 250 ps interlaced, 1 - 250 ps interpolated,
                                    # 2 - 500 ps, 3 - 1 ns, 4 - 2 ns
            'TimeOffset',           # delay post trigger in us
            'VerticalOffset'        # -0.25 V to 0.25 V in steps of 30 uV
        ]

        self.general_settings = [
            'ActiveProtoNumber',        # Which protocol to use for acquisition
            'ExtTriggerInputEdge',      # 0 - rising, 1 - falling
            'ExtTriggerInputEnable',    # 1 - enabled, 0 - disabled
            'ExtTriggerInputThreshold', # -2.5 V to 2.5 V in 0.01 V steps
            'PresetFlags',              # 1 - on, 0 - off
            'Rps',                      # rapid protocol selection 1 - on, 0 - off
            'SpectrumPreset',           # number of preset spectra for acquisition
            'TimePreset',               # preset acquisition time in seconds
            'TriggerEnablePolarity',    # 0 - TTL low, 1 - TTL high
            'TriggerOutputWidth'        # 0.064 us to 8.192 us in units of us
        ]

        self.tof_parms = [
            'ErrFlags',
            'ProtoNum',
            'SpecNum',
            'TimeStamp'
        ]

    def __del__(self):
        if self.is_connected():
            self.close()
    
    def get_prot_parms(self, prot_num: int):
        """
        Convert ProtocolObj to dictionary of relevant parameters.
        """
        if prot_num not in range(16):
            raise IndexError("Protocol number must be 0-15.")

        return {parm: getattr(self.Protocols[prot_num], parm, None) 
                for parm in self.protocol_parms}
    
    def set_prot_parms(self, prot_num: int, **kwargs):
        """
        Load a ProtocolObj in self.Protocols with the proper 
        parameters.
        """
        if prot_num not in range(16):
            raise IndexError("Protocol number must be 0-15.")

        for parm in self.protocol_parms:
            arg = kwargs.get(parm)
            if arg is not None:
                setattr(self.Protocols[prot_num], parm, arg)

    def get_protocol(self, prot_num: int):
        """
        Retrieve a ProtocolObj from the FastFlight. 
        Return a dictionary of relevant parameters.
        """
        if prot_num not in range(16):
            raise IndexError("Protocol number must be 0-15.")
        
        self.FF2Ctrl.GetProtocol(prot_num, self.Protocols[prot_num])
        return self.get_prot_parms(prot_num)
    
    def set_protocol(self, prot_num: int, **kwargs):
        """Set a protocol on the FastFlight"""
        if prot_num not in range(16):
            raise IndexError("Protocol number must be 0-15.")

        self.set_prot_parms(prot_num, **kwargs)
        self.FF2Ctrl.SetProtocol(prot_num, self.Protocols[prot_num])
        self.dither_ready = False

    def get_gs_parms(self):
        """
        Convert GSObj to a dict of relevant parameters.
        """
        return {parm: getattr(self.GSObj, parm, None) 
                for parm in self.general_settings}
    
    def set_gs_parms(self, **kwargs):
        """
        Load GSObj with relevant parameters.
        """
        for parm in self.general_settings:
            arg = kwargs.get(parm)
            if arg is not None:
                setattr(self.GSObj, parm, arg)

    def get_general_settings(self):
        """
        Retrieve GSObj from the FastFlight.
        Return a dictionary of relevant parameters.
        """
        self.FF2Ctrl.GetGeneralSettings(self.GSObj)
        return self.get_gs_parms()
    
    def set_general_settings(self, **kwargs):
        """Set general settings on the FastFlight."""
        self.set_gs_parms(**kwargs)
        self.FF2Ctrl.SetGeneralSettings(self.GSObj)

    def get_tof_parms(self):
        """Return a dictionary describing TOFObj."""
        result = {}
        for parm in self.tof_parms:
            value = getattr(self.TOFObj, parm, None)
            if value is not None:
                # Ensure integer types are properly converted
                if parm in ['ErrFlags', 'ProtoNum', 'SpecNum', 'TagNum']:
                    # Explicitly convert to python int for consistent bit repr
                    result[parm] = int(value)
                else:
                    # Double precision values 
                    # (SpecificIonCount, TimeStamp, TotalIonCount)
                    result[parm] = float(value)
            else:
                result[parm] = None    
        return result
    
    def is_acq_running(self) -> bool | None:
        """Is there a currently running acquisition."""
        return self.FF2Ctrl.Active

    def device_count(self) -> int | None:
        """Number of devices available for USB connection."""
        return self.FF2Ctrl.DeviceCount
    
    def FF_version(self) -> str | None:
        """Version of the FastFlight firmware."""
        return self.FF2Ctrl.FFVersion
    
    def is_connected(self) -> bool | None:
        """Is there an open connection to instrument."""
        return self.FF2Ctrl.IsInstrumentPresent
    
    def num_records(self) -> int | None:
        """Records collected in current acquisition."""
        return self.FF2Ctrl.Records
    
    def serial_number(self) -> str | None:
        """Serial number of the FastFlight."""
        return self.FF2Ctrl.SerialNumber
    
    def num_spectra(self) -> int | None:
        """Number of spectra collected in acquisition."""
        return self.FF2Ctrl.Spectrums
    
    def time_elapsed(self) -> float | None:
        """Seconds since start of acquisition."""
        return self.FF2Ctrl.TimeElapsed
    
    def open(self) -> bool:
        """
        Open connection to the FastFlight.
        Returns True when successful.
        """
        return self.FF2Ctrl.Open()

    def close(self):
        """Close connection to the FastFlight."""
        self.FF2Ctrl.Close()

    def reset_time_stamp(self):
        """Reset time stamp clock on FastFlight hardware."""
        self.FF2Ctrl.ResetTimeStamp()

    def start_acq(self, reset_time_stamp: bool = False):
        """
        Start an acquisition.
        We only perform acquisitions in TOF mode.
        """
        self.FF2Ctrl.Start(0, int(reset_time_stamp))

    def stop_acq(self):
        """Stop the current acquisition."""
        self.FF2Ctrl.Stop()

    def get_data(self) -> Tuple[list, list, dict] | None:
        """Get TOF data off the FastFlight."""
        vaXData = VARIANT()
        vaYData = VARIANT()
        success = self.FF2Ctrl.GetTOFData(vaXData, vaYData, 
                                          self.TOFObj)
        if not success:
            return
        return vaXData.value, vaYData.value, self.get_tof_parms()
    
    def _wait_for_new_spectrum(self, baseline: int):
        prot = self.Protocols[self.GSObj.ActiveProtoNumber]
        rec_len = 1e-6 * float(prot.RecordLength)
        target_rec = float(prot.RecordsPerSpectrum)
        wait_step = max(target_rec * rec_len * 0.1, 0.01)

        timeout = time.time() + self._acq_timeout
        while True:
            current = self.num_spectra() or 0
            if current > baseline:
                return
            if time.time() > timeout:
                raise TimeoutError("Timed out waiting for new spectrum.")
            if not self.is_acq_running():
                raise RuntimeError("Acquisition unexpectedly stopped.")
            time.sleep(wait_step)


    def get_spectrum(self) -> Tuple[list, list, dict] | None:
        baseline = self.num_spectra() or 0

        if not self.is_acq_running():
            self.start_acq()

        self._wait_for_new_spectrum(baseline)
        return self.get_data()

    
    def get_num_spectra_per_trace(self) -> int:
        return self._num_spectra_per_trace
    
    def set_num_spectra_per_trace(self, N: int):
        self._num_spectra_per_trace = N
    
    def get_trace(self) -> Tuple[list, list, dict] | None:
        self.start_acq()
        try:
            T, V, D = self.get_spectrum()
            V = list(V)

            for _ in range(self._num_spectra_per_trace - 1):
                t, v, d = self.get_spectrum()
                for i in range(len(t)):
                    V[i] += v[i]
                D['ErrFlags'] |= d['ErrFlags']

            prot = self.Protocols[self.GSObj.ActiveProtoNumber]
            D['SpecNum'] = self._num_spectra_per_trace * prot.RecordsPerSpectrum

            return T, V, D
        finally:
            self.stop_acq()

    
    def get_dither_len(self) -> float:
        return self.dither_len

    def prep_dither(self, dither_len: float):
        """Prepare protocols for taking a dithered trace."""
        prot_i = self.GSObj.ActiveProtoNumber
        proto_args = self.get_prot_parms(prot_i)
        delay = proto_args['TimeOffset']

        for i in range(16):
            proto_args['TimeOffset'] = delay + i * dither_len / 15
            self.set_protocol(i, **proto_args)

        self.set_general_settings(ActiveProtoNumber = 0)
        self.dither_len = dither_len
        self.dither_ready = True
        
def get_trace_dither(self) -> Tuple[list, list, dict] | None:
    if not self.dither_ready:
        raise RuntimeError("Protocols are not prepped for dithering.")

    self.start_acq()
    try:
        for i in range(self._num_spectra_per_trace):
            self.set_general_settings(ActiveProtoNumber=i % 16)

            if i == 0:
                T, V, D = self.get_spectrum()
                V = list(V)
            else:
                t, v, d = self.get_spectrum()
                voff = self.Protocols[0].VerticalOffset
                num_avg = self.Protocols[0].RecordsPerSpectrum
                for j in range(len(t)):
                    offset = self.Protocols[i % 16].VerticalOffset - voff
                    offset_int = int(offset * num_avg * 256 / 0.5)
                    V[j] += v[j] - offset_int
                D['ErrFlags'] |= d['ErrFlags']

        D['SpecNum'] = self._num_spectra_per_trace * num_avg
        return T, V, D
    finally:
        self.stop_acq()
