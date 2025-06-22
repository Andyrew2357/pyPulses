import json
import sys
import traceback
from typing import Any, Dict

################################################################################
"""Copy of fastflight32 (imports get tricky with the subprocess)"""

from comtypes.client import CreateObject # type: ignore
from comtypes.automation import VARIANT # type: ignore
from typing import Optional, Tuple

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
        return {parm: getattr(self.TOFObj, parm, None)
                for parm in self.tof_parms}
    
    def is_acq_running(self) -> Optional[bool]:
        """Is there a currently running acquisition."""
        return self.FF2Ctrl.Active

    def device_count(self) -> Optional[int]:
        """Number of devices available for USB connection."""
        return self.FF2Ctrl.DeviceCount
    
    def FF_version(self) -> Optional[str]:
        """Version of the FastFlight firmware."""
        return self.FF2Ctrl.FFVersion
    
    def is_connected(self) -> Optional[bool]:
        """Is there an open connection to instrument."""
        return self.FF2Ctrl.IsInstrumentPresent
    
    def num_records(self) -> Optional[int]:
        """Records collected in current acquisition."""
        return self.FF2Ctrl.Records
    
    def serial_number(self) -> Optional[str]:
        """Serial number of the FastFlight."""
        return self.FF2Ctrl.SerialNumber
    
    def num_spectra(self) -> Optional[int]:
        """Number of spectra collected in acquisition."""
        return self.FF2Ctrl.Spectrums
    
    def time_elapsed(self) -> Optional[float]:
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

    def get_data(self) -> Optional[Tuple[list, list, dict]]:
        """Get TOF data off the FastFlight."""
        vaXData = VARIANT()
        vaYData = VARIANT()
        success = self.FF2Ctrl.GetTOFData(vaXData, vaYData, 
                                          self.TOFObj)
        if not success:
            return
        return vaXData.value, vaYData.value, self.get_tof_parms()

################################################################################

class FastFlightBridge():
    def __init__(self):
        self.ff2 = None

    def handle_request(self, request: Dict[str, Any]):
        """Handle a single request and return response"""

        try:
            method = request['method']
            args = request.get('args', [])
            kwargs = request.get('kwargs', {})

            # Initialize FastFlight instance if needed
            if self.ff2 is None and method != 'init':
                self.ff2 = FastFlight32()

            # Handle method calls
            if method == 'init':
                self.ff2 = FastFlight32()
                return {'success': True, 'result': None}
                
            elif method == 'del':
                if self.ff2:
                    del self.ff2
                    self.ff2 = None
                return {'success': True, 'result': None}
            
            elif hasattr(self.ff2, method):
                result = getattr(self.ff2, method)(*args, **kwargs)
                return {'success': True, 'result': result}
            
            else:
                return {'success': False, 
                        'error'  : f'Unknown method: {method}'
                    }
            
        except Exception as e:
            return {
                'success'   : False,
                'error'     : str(e),
                'traceback' : traceback.format_exc()
            }
        
    def run(self):
        """
        Main message loop. Read JSON requests from stdin, write to stdout.
        """

        try:
            while True:
                line = sys.stdin.readline().strip()
                
                if not line or line == 'QUIT':
                    break

                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response), flush = True)

                except json.JSONDecodeError as e:
                    error_response = {
                        'success': False, 
                        'error'  : f'JSON decode error: {str(e)}'
                    }
                    print(json.dumps(error_response), flush = True)

        except KeyboardInterrupt:
            pass
        
        finally:
            if self.ff2:
                try:
                    if self.ff2.is_connected():
                        self.ff2.close()
                except:
                    pass

if __name__ == '__main__':
    bridge = FastFlightBridge()
    bridge.run()
