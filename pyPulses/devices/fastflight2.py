from .fastflight2_utils import FastFlight64
from .fastflight_scopeview import FFScopeView
from .abstract_device import abstractDevice
from ._registry import DeviceRegistry

import numpy as np
from typing import Any, Tuple

class FastFlight2(abstractDevice):
    def __init__(self, logger = None):
        super().__init__(logger)
        self.ff2 = FastFlight64()
        DeviceRegistry.register_device('FASTFLIGHT2', self)

        self.voltage_scale_factor = 1.0

        # Mapping dictionaries
        self._compression_modes = {
            'lossy': 0, 'lossless': 1, 'stick': 2
        }
        self._compression_modes_inv = {v: k for k, v in self._compression_modes.items()}
        
        self._time_resolutions = {
            '250ps_interlaced': 0, '250ps_interpolated': 1, 
            '500ps': 2, '1ns': 3, '2ns': 4
        }
        self._time_resolutions_inv = {v: k for k, v in self._time_resolutions.items()}

        # Parameter metadata
        self._prot_param_cache = {
            'compression'                 : ('CompType', self._compression_modes, self._compression_modes_inv),
            'correlated_noise_subtraction': ('EnableCNS', None, None),
            'precision_enhancement'       : ('PrecEnhEnable', None, None),
            'trace_length'                : ('RecordLength', 1.0e6, 0, 3.0e-3), # scale, min, max
            'num_averages'                : ('RecordsPerSpectrum', 1, 1, 65535),
            'time_resolution'             : ('SamplingInterval', self._time_resolutions, self._time_resolutions_inv),
            'trigger_delay'               : ('TimeOffset', 1.0e6, 0, 1.0e-3),
            'voltage_offset'              : ('VerticalOffset', 1.0, -0.25, 0.25)
        }
        
        self._gen_param_cache = {
            'active_protocol'        : ('ActiveProtoNumber', 1, 0, 15),
            'trigger_falling'        : ('ExtTriggerInputEdge', None, None),
            'trigger_enabled'        : ('ExtTriggerInputEnable', None, None),
            'trigger_threshold'      : ('ExtTriggerInputThreshold', 1.0, -2.5, 2.5),
            'trigger_enable_polarity': ('TriggerEnablePolarity', None, None),
            'trigger_output_width'   : ('TriggerOutputWidth', 1.0e6, 0.064e-6, 8.192e-6)
        }

        # Initialize settings storage
        self.protocols = [self.ff2.get_prot_parms(i) for i in range(15)]
        self.gen_settings = self.ff2.get_gs_parms()
        
        # Settings sync flag to avoid unnecessary retrievals
        self._settings_synced = False

    def __del__(self):
        del self.ff2
        super().__del__()

    def _ensure_settings_synced(self):
        """Only sync settings if needed"""
        if not self._settings_synced:
            self._sync_machine_settings()
            self._settings_synced = True

    def _invalidate_settings_cache(self):
        """Mark settings as needing resync"""
        self._settings_synced = False

    def _sync_machine_settings(self):
        """Synchronize protocol and general settings with the device"""
        for i in range(15):
            self.protocols[i] = self.ff2.get_protocol(i)
        self.gen_settings = self.ff2.get_general_settings()

    def _log_settings(self):
        """Log current settings - simplified version"""
        msg = "RETRIEVED SETTINGS FROM DEVICE\n"
        # Only log active protocol to reduce overhead
        active_prot = self.gen_settings.get('ActiveProtoNumber', 0)
        msg += f"ACTIVE PROTOCOL {active_prot}:\n"
        
        for param_name, (hw_name, *_) in self._prot_param_cache.items():
            value = self.protocols[active_prot].get(hw_name)
            if param_name == 'compression':
                value = self._compression_modes_inv.get(value, value)
            elif param_name == 'time_resolution':
                value = self._time_resolutions_inv.get(value, value)
            elif param_name in ['correlated_noise_subtraction', 
                                'precision_enhancement']:
                value = bool(value)
            elif param_name in ['trace_length', 'trigger_delay']:
                value = value / 1.0e6 if value is not None else None
            msg += f"{param_name:>24} = {value:>16}\n"
        
        self.info(msg)

    # Parameter access methods
    def _get_protocol_param(self, param_name: str, prot_num: int = None) -> Any:
        """Fast protocol parameter getter"""
        self._ensure_settings_synced()
        prot_num = prot_num or self.gen_settings['ActiveProtoNumber']
        
        hw_name, *meta = self._prot_param_cache[param_name]
        raw_value = self.protocols[prot_num][hw_name]
        
        # Handle different parameter types
        if param_name == 'compression':
            return self._compression_modes_inv[raw_value]
        elif param_name == 'time_resolution':
            return self._time_resolutions_inv[raw_value]
        elif param_name in ['correlated_noise_subtraction', 
                            'precision_enhancement']:
            return bool(raw_value)
        elif param_name in ['trace_length', 'trigger_delay']:
            return raw_value / 1.0e6
        elif param_name == 'num_averages':
            return int(raw_value)
        else:
            return raw_value

    def _set_protocol_param(self, param_name: str, value: Any, 
                            prot_num: int = None):
        """Fast protocol parameter setter"""
        prot_num = prot_num or self.gen_settings['ActiveProtoNumber']
        hw_name, *meta = self._prot_param_cache[param_name]
        
        # Convert value to hardware format
        if param_name == 'compression':
            if value not in self._compression_modes:
                raise ValueError(f"Invalid compression mode: {value}")
            hw_value = self._compression_modes[value]
        elif param_name == 'time_resolution':
            if value not in self._time_resolutions:
                raise ValueError(f"Invalid time resolution: {value}")
            hw_value = self._time_resolutions[value]
        elif param_name in ['correlated_noise_subtraction', 'precision_enhancement']:
            hw_value = int(bool(value))
        elif param_name in ['trace_length', 'trigger_delay']:
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = value * scale
        elif param_name == 'num_averages':
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = int(value)
        elif param_name == 'voltage_offset':
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = value
        else:
            hw_value = value

        # Update local cache and push to device
        self.ff2.set_prot_parms(prot_num, **{hw_name: hw_value})
        self.ff2.set_protocol(prot_num)
        
        # Update local cache
        self.protocols[prot_num][hw_name] = hw_value
        self.info(f"Setting: {param_name} = {value}")

    def _get_general_param(self, param_name: str) -> Any:
        """Fast general parameter getter"""
        self._ensure_settings_synced()
        hw_name, *meta = self._gen_param_cache[param_name]
        raw_value = self.gen_settings[hw_name]
        
        if param_name in ['trigger_falling', 
                          'trigger_enabled', 
                          'trigger_enable_polarity']:
            return bool(raw_value)
        elif param_name == 'trigger_output_width':
            return raw_value / 1.0e6
        else:
            return raw_value

    def _set_general_param(self, param_name: str, value: Any):
        """Fast general parameter setter"""
        hw_name, *meta = self._gen_param_cache[param_name]
        
        # Convert and validate
        if param_name in ['trigger_falling', 
                          'trigger_enabled', 
                          'trigger_enable_polarity']:
            hw_value = int(bool(value))
        elif param_name in ['active_protocol']:
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = int(value)
        elif param_name in ['trigger_threshold']:
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = value
        elif param_name == 'trigger_output_width':
            scale, min_val, max_val = meta[0], meta[1], meta[2]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
            hw_value = value * scale
        else:
            hw_value = value

        # Update device and local cache
        self.ff2.set_gs_parms(**{hw_name: hw_value})
        self.ff2.set_general_settings()
        self.gen_settings[hw_name] = hw_value
        self.info(f"Setting: {param_name} = {value}")

    # Protocol Parameters
    def compression(self, mode: str = None, prot_num: int = None) -> str:
        if mode is not None:
            self._set_protocol_param('compression', mode, prot_num)
        return self._get_protocol_param('compression', prot_num)
    
    def correlated_noise_subtraction(self, on: bool = None, 
                                     prot_num: int = None) -> bool:
        if on is not None:
            self._set_protocol_param('correlated_noise_subtraction', on, prot_num)
        return self._get_protocol_param('correlated_noise_subtraction', prot_num)
    
    def precision_enhancement(self, on: bool = None, prot_num: int = None
                              ) -> bool:
        if on is not None:
            self._set_protocol_param('precision_enhancement', on, prot_num)
        return self._get_protocol_param('precision_enhancement', prot_num)
    
    def trace_length(self, val: float = None, prot_num: int = None) -> float:
        if val is not None:
            self._set_protocol_param('trace_length', val, prot_num)
        return self._get_protocol_param('trace_length', prot_num)
    
    def num_averages(self, n: int = None, prot_num: int = None) -> int:
        if n is not None:
            self._set_protocol_param('num_averages', n, prot_num)
        return self._get_protocol_param('num_averages', prot_num)

    def time_resolution(self, setting: str = None, prot_num: int = None) -> str:
        if setting is not None:
            self._set_protocol_param('time_resolution', setting, prot_num)
        return self._get_protocol_param('time_resolution', prot_num)

    def trigger_delay(self, val: float = None, prot_num: int = None) -> float:
        if val is not None:
            self._set_protocol_param('trigger_delay', val, prot_num)
        return self._get_protocol_param('trigger_delay', prot_num)

    def voltage_offset(self, val: float = None, prot_num: int = None) -> float:
        if val is not None:
            self._set_protocol_param('voltage_offset', val, prot_num)
        return self._get_protocol_param('voltage_offset', prot_num)

    # General Settings Parameters
    def active_protocol(self, prot_num: int = None) -> int:
        if prot_num is not None:
            self._set_general_param('active_protocol', prot_num)
        return self._get_general_param('active_protocol')
    
    def trigger_falling(self, falling: bool = None) -> bool:
        if falling is not None:
            self._set_general_param('trigger_falling', falling)
        return self._get_general_param('trigger_falling')
    
    def trigger_enabled(self, on: bool = None) -> bool:
        if on is not None:
            self._set_general_param('trigger_enabled', on)
        return self._get_general_param('trigger_enabled')
    
    def trigger_threshold(self, val: float = None) -> float:
        if val is not None:
            self._set_general_param('trigger_threshold', val)
        return self._get_general_param('trigger_threshold')
    
    def trigger_enable_polarity(self, high: bool = None) -> bool:
        if high is not None:
            self._set_general_param('trigger_enable_polarity', high)
        return self._get_general_param('trigger_enable_polarity')
    
    def trigger_output_width(self, val: float = None) -> float:
        if val is not None:
            self._set_general_param('trigger_output_width', val)
        return self._get_general_param('trigger_output_width')
    
    # Bulk parameter setting
    def set_protocol_bulk(self, prot_num: int = None, **kwargs):
        """
        Set multiple protocol parameters in one call
        """
        prot_num = prot_num or self.gen_settings['ActiveProtoNumber']
        hw_params = {}
        
        for param_name, value in kwargs.items():
            if param_name in self._prot_param_cache:
                hw_name, *meta = self._prot_param_cache[param_name]
                # Apply same conversion logic as individual setters
                if param_name == 'compression':
                    hw_params[hw_name] = self._compression_modes[value]
                elif param_name == 'time_resolution':
                    hw_params[hw_name] = self._time_resolutions[value]
                elif param_name in ['correlated_noise_subtraction', 
                                    'precision_enhancement']:
                    hw_params[hw_name] = int(bool(value))
                elif param_name in ['trace_length', 'trigger_delay']:
                    hw_params[hw_name] = value * meta[0]
                elif param_name == 'num_averages':
                    hw_params[hw_name] = int(value)
                else:
                    hw_params[hw_name] = value
        
        if hw_params:
            self.ff2.set_prot_parms(prot_num, **hw_params)
            self.ff2.set_protocol(prot_num)
            # Update local cache
            self.protocols[prot_num].update(hw_params)
            self.info(f"Bulk protocol update: {kwargs}")

    def set_general_bulk(self, **kwargs):
        """Set multiple general parameters in one call"""
        hw_params = {}
        
        for param_name, value in kwargs.items():
            if param_name in self._gen_param_cache:
                hw_name, *meta = self._gen_param_cache[param_name]
                # Apply conversion logic
                if param_name in ['trigger_falling', 
                                  'trigger_enabled', 
                                  'trigger_enable_polarity']:
                    hw_params[hw_name] = int(bool(value))
                elif param_name == 'trigger_output_width':
                    hw_params[hw_name] = value * meta[0]
                else:
                    hw_params[hw_name] = value
        
        if hw_params:
            self.ff2.set_gs_parms(**hw_params)
            self.ff2.set_general_settings()
            self.gen_settings.update(hw_params)
            self.info(f"Bulk general settings update: {kwargs}")

    # Keep the rest of the methods unchanged
    def is_acq_running(self) -> bool:
        return self.ff2.is_acq_running()
    
    def device_count(self) -> int:
        return self.ff2.device_count()
    
    def is_connected(self) -> bool:
        return self.ff2.is_connected()
    
    def num_records(self) -> int:
        return self.ff2.num_records()
    
    def time_elapsed(self) -> float:
        return self.ff2.time_elapsed()

    def connect(self):
        if not self.is_connected():
            self.ff2.open()
            self.info("FF2 connected.")
            self._sync_machine_settings()
            self._settings_synced = True
            self._log_settings()

    def disconnect(self):
        if self.is_connected():
            if self.is_acq_running():
                self.info("Requested closure with acquisition running; Ending acquisition...")
                self.stop_acq()
            self.ff2.close()
            self.info("FF2 disconnected.")
            self._invalidate_settings_cache()

    def start_acq(self):
        self.info(f"Manually Starting Acquisition")
        self.ff2.start_acq()

    def stop_acq(self):
        self.ff2.stop_acq()
        self.info(f"Manually Stopped Acquisition")

    def prep_dither(self, dither_len: float):
        """Set up protocols with an appropriate dither length"""
        self.ff2.prep_dither(dither_len)
        self.info(
            f"Prepared protocols for dithering; dither length = {dither_len} V"
            f"Resyncing cached protocols and settings..."
        )
        self._invalidate_settings_cache()
        self._ensure_settings_synced()
        self._log_settings()

    def get_dither_length(self) -> float:
        return self.ff2.get_dither_len()

    def spectra_per_trace(self, N: int = None) -> int:
        """
        Number of distinct spectra to take in a single trace 
        (irrespective of whether we are dithering)
        """

        if N is None:
            return self.ff2.get_num_spectra_per_trace()
        self.ff2.set_num_spectra_per_trace(N)
        self.info(f"Set to take {N} spectra per trace")
        return N

    def get_trace(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Take a trace with the fastflight"""

        T, V, D = self.ff2.get_trace()
        N = D['SpecNum'] # we fill this with num_avg * records per spectrum
        V *= self.voltage_scale_factor * 0.5 / (256 * N)
        errflags = D['ErrFlags']
        self.parse_error_flags(errflags)
        self.info(f"Took trace with {len(T)} points, {N} averages.")
        return T, V, errflags

    def get_trace_dither(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Take a trace with the fastflight"""

        T, V, D = self.ff2.get_trace_dither()
        N = D['SpecNum'] # we fill this with num_avg * records per spectrum
        V *= self.voltage_scale_factor * 0.5 / (256 * N)
        errflags = D['ErrFlags']
        self.parse_error_flags(errflags)
        self.info(f"Took a dithered trace with {len(T)} points, {N} averages.")
        return T, V, errflags

    def parse_error_flags(self, errflags):
        """Appropriately log the results of the error flag"""
        self.info(
            f"Error Flags: ADC underflow = {errflags & 1}\n"
                         f"ADC overflow  = {errflags & 2}"
        )

    def launch_scope_view(self):
        """Launch a GUI to use the fastflight as an Oscilloscope"""
        self.info("LAUNCHING SCOPE VIEW APPLICATION...")
        self.disconnect()
        FFScopeView(self.ff2)
        self.info("CLOSED SCOPE VIEW APPLICATION WINDOW\nReconnecting...")
        self.connect()
