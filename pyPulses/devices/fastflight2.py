from .fastflight2_utils import FastFlight64
from .fastflight_scopeview import FFScopeView
from .abstract_device import abstractDevice
from ._registry import DeviceRegistry

class FastFlight2(abstractDevice):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.ff2 = FastFlight64()
        DeviceRegistry.register_device('FASTFLIGHT2', self)

        self.compression_mode = {
            'lossy'     : 0,
            'lossless'  : 1,
            'stick'     : 2
        }

        self.protocols = [None] * 16
        self.general_settings = {}

    def connect(self):
        if not self.ff2.is_connected():
            self.ff2.open()
            self.info("FF2 connected.")

    def disconnect(self):
        if self.ff2.is_connected():
            self.ff2.close()
            self.info("FF2 disconnected.")

    def prep_protocol(self,
                      prot_num: int     = None,
                      compression: str  = None,
                      cns: bool         = None,
                      prec_enh: bool    = None,
                      rec_len: float    = None,
                      averages: int     = None,
                      time_off: float   = None,
                      input_off: float  = None) -> dict:
        
        prot_parms = {}
        if compression is self.compression_mode: 
            prot_parms['CompType'] = self.compression_mode[compression]
        if cns is not None:
            prot_parms['EnableCNS'] = int(cns)
        if prec_enh is not None:
            prot_parms['PrecEnhEnable'] = int(prec_enh)
        if rec_len is not None and (0 <= rec_len < 3e-3):
            pass

        # FINISH THIS...
        

    def launch_scope_view(self):
        """
        Launch a GUI to use the fastflight as an Oscilloscope
        """
        FFScopeView(self.ff2)

# self.protocol_parms = [
#             'CompType',             # 0 - lossy, 1 - lossless, 2 - stick (unused)
#             'EnableCNS',            # 1 - enabled, 0 - disabled
#             'PrecEnhEnable',        # 1 - enabled, 0 - disabled
#             'RecordLength',         # in units of us
#             'RecordsPerSpectrum',   # between 1 and 65_535
#             'SamplingInterval',     # 0 - 250 ps interlaced, 1 - 250 ps interpolated,
#                                     # 2 - 500 ps, 3 - 1 ns, 4 - 2 ns
#             'TimeOffset',           # delay post trigger in us
#             'VerticalOffset'        # -0.25 V to 0.25 V in steps of 30 uV
#         ]

#         self.general_settings = [
#             'ActiveProtoNumber',        # Which protocol to use for acquisition
#             'ExtTriggerInputEdge',      # 0 - rising, 1 - falling
#             'ExtTriggerInputEnable',    # 1 - enabled, 0 - disabled
#             'ExtTriggerInputThreshold', # -2.5 V to 2.5 V in 0.01 V steps
#             'PresetFlags',              # 1 - on, 0 - off
#             'Rps',                      # rapid protocol selection 1 - on, 0 - off
#             'SpectrumPreset',           # number of preset spectra for acquisition
#             'TimePreset',               # preset acquisition time in seconds
#             'TriggerEnablePolarity',    # 0 - TTL low, 1 - TTL high
#             'TriggerOutputWidth'        # 0.064 us to 8.192 us in units of us
#         ]

#         self.tof_parms = [
#             'ErrFlags',
#             'ProtoNum',
#             'SpecNum',
#             'TimeStamp'
#         ]