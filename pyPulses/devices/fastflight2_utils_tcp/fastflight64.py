from .ff_package_manager import FastFlightPackageManager

import socket
import json
import struct
import subprocess
import time
import threading
import numpy as np
from typing import Tuple

class FastFlight64():
    def __init__(self, host='127.0.0.1', port=50432, server_cmd=None):
        self.host = host
        self.port = port

        if server_cmd is None:
            package_manager = FastFlightPackageManager()
            python32_path = package_manager.get_python32_path()
            bridge_script_path = package_manager.get_bridge_script_path()
            self.server_cmd = server_cmd or [python32_path, bridge_script_path,
                                             host, str(port)]
        else:
            self.server_cmd = server_cmd

        self._lock = threading.Lock()
        self._connect_or_start()

    def _connect_or_start(self):
        try:
            self._connect()
        except (ConnectionRefusedError, OSError):
            print("Starting TCP bridge server...")
            self._server_proc = subprocess.Popen(self.server_cmd)
            time.sleep(1.5)
            self._connect()

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.rfile = self.sock.makefile('rb')
        self.wfile = self.sock.makefile('wb')

    def _call_method(self, method, *args, **kwargs):
        request = {'method': method, 'args': list(args), 'kwargs': kwargs}
        
        with self._lock:
            self.wfile.write((json.dumps(request) + '\n').encode('utf-8'))
            self.wfile.flush()

            header = self.rfile.readline()
            if header == b'BINARY_DATA_FOLLOWS\n':
                # Read fixed-size metadata: sampling_interval(4) + err_flags(4) + 
                # proto_num(4) + spec_num(4) + timestamp(8) = 24 bytes
                meta = self._read_exact(24)
                sampling_interval, err_flags, proto_num, spec_num = \
                    struct.unpack('<IIII', meta[:16])
                timestamp = struct.unpack('<d', meta[16:24])[0]
                print(sampling_interval, err_flags, proto_num, spec_num, timestamp)
                n = struct.unpack('<I', self._read_exact(4))[0] # data length
                y_data = np.frombuffer(self._read_exact(4 * n), dtype = '<i4') # 4 bytes per little-endian int32
                x_data = self._reconstruct_x_data(n, sampling_interval)
                return x_data, y_data.astype(float), {
                    'ErrFlags': err_flags,
                    'ProtoNum': proto_num,
                    'SpecNum': spec_num,
                    'TimeStamp': timestamp
                }
            
            response = json.loads(header.decode('utf-8'))
            if response['success']:
                return response['result']
            
            raise RuntimeError(response['error'] + '\n' + \
                               response.get('traceback', ''))
        
    def _read_exact(self, n_bytes: int) -> bytes:
        """Read exactly n_bytes from the socket, handling partial reads."""
        data = b''
        while len(data) < n_bytes:
            chunk = self.rfile.read(n_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data
        
    def _reconstruct_x_data(self, length, sampling_interval):
        interval_map = {0: 0.25, 1: 0.25, 2: 0.5, 3: 1.0, 4: 2.0}
        return interval_map[sampling_interval] * np.arange(length)
    
    def __del__(self):
        """Clean up the bridge process"""
        if self.is_connected():
            self.close()

        try:
            self.sock.close()
        except:
            pass

    """Mirror all FastFlight32 methods"""
    def get_prot_parms(self, prot_num: int):
        return self._call_method('get_prot_parms', prot_num)
    
    def set_prot_parms(self, prot_num: int, **kwargs):
        return self._call_method('set_prot_parms', prot_num, **kwargs)
    
    def get_protocol(self, prot_num: int):
        return self._call_method('get_protocol', prot_num)
    
    def set_protocol(self, prot_num: int, **kwargs):
        return self._call_method('set_protocol', prot_num, **kwargs)
    
    def get_gs_parms(self):
        return self._call_method('get_gs_parms')
    
    def set_gs_parms(self, **kwargs):
        return self._call_method('set_gs_parms', **kwargs)
    
    def get_general_settings(self):
        return self._call_method('get_general_settings')
    
    def set_general_settings(self, **kwargs):
        return self._call_method('set_general_settings', **kwargs)
    
    def get_tof_parms(self):
        return self._call_method('get_tof_parms')
    
    def is_acq_running(self) -> bool | None:
        return self._call_method('is_acq_running')
    
    def device_count(self) -> int | None:
        return self._call_method('device_count')
    
    def FF_version(self) -> str | None:
        return self._call_method('FF_version')
    
    def is_connected(self) -> bool | None:
        return self._call_method('is_connected')
    
    def num_records(self) -> int | None:
        return self._call_method('num_records')
    
    def serial_number(self) -> str | None:
        return self._call_method('serial_number')
    
    def num_spectra(self) -> int | None:
        return self._call_method('num_spectra')
    
    def time_elapsed(self) -> float | None:
        return self._call_method('time_elapsed')
    
    def open(self) -> bool:
        return self._call_method('open')
    
    def close(self):
        return self._call_method('close')
    
    def reset_time_stamp(self):
        return self._call_method('reset_time_stamp')
    
    def start_acq(self, reset_time_stamp: bool = False):
        return self._call_method('start_acq', reset_time_stamp)
    
    def stop_acq(self):
        return self._call_method('stop_acq')
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, dict] | None:
        """
        Get TOF data from the FastFlight.
        
        Returns:
            Tuple of (x_data, y_data, tof_params) or None if no data available
            
            x_data: list of float - Time values in microseconds
            y_data: list of float - Signal intensity values (converted from integers)
            tof_params: dict containing:
                - ErrFlags (int): Error status bits (0=ADC underflow, 1=ADC overflow)
                        Check (ErrFlags & 1) for ADC underflow,
                              (ErrFlags & 2) for ADC overflow
                - ProtoNum (int): Protocol number used for this spectrum
                - SpecificIonCount (float): Specific ion count for this spectrum
                - SpecNum (int): Spectrum number since acquisition start
                - TagNum (int): Tag number for this spectrum  
                - TimeStamp (float): Timestamp in seconds
                - TotalIonCount (float): Total ion count for this spectrum
        """

        result = self._call_method('get_data')
        if result is None:
            return None
        return tuple(result)
    
    def get_num_spectra_per_trace(self) -> int:
        """Get the number of spectra to include in a single trace"""
        return self._call_method('get_num_spectra_per_trace')
    
    def set_num_spectra_per_trace(self, N: int):
        """Set the number of spectra to include in a single trace"""
        return self._call_method('set_num_spectra_per_trace', N)
    
    def get_trace(self) -> Tuple[np.ndarray, np.ndarray, dict] | None:
        """
        Get a trace from the FastFlight.
        
        Returns:
            Tuple of (x_data, y_data, tof_params) or None if no data available
            
            x_data: list of float - Time values in microseconds
            y_data: list of float - Signal intensity values (converted from integers)
            tof_params: dict containing:
                - ErrFlags (int): Error status bits (0=ADC underflow, 1=ADC overflow)
                        Check (ErrFlags & 1) for ADC underflow,
                              (ErrFlags & 2) for ADC overflow
                - ProtoNum (int): Protocol number used for this spectrum
                - SpecificIonCount (float): Specific ion count for this spectrum
                - SpecNum (int): REPLACED WITH num_spectra_per_trace * RecordsPerSpectrum
                - TagNum (int): Tag number for this spectrum  
                - TimeStamp (float): Timestamp in seconds
                - TotalIonCount (float): Total ion count for this spectrum
        """

        result = self._call_method('get_trace')
        if result is None:
            return None
        return tuple(result)
    
    def get_dither_len(self) -> float:
        """Get the dither length (V)"""
        return self._call_method('get_dither_len')

    def prep_dither(self, dither_len: float):
        """Set the dither length and prep protocols for a dithered trace"""
        return self._call_method('prep_dither', dither_len)
    
    def get_trace_dither(self) -> Tuple[np.ndarray, np.ndarray, dict] | None:
        """
        Get a dithered trace from the FastFlight.
        
        Returns:
            Tuple of (x_data, y_data, tof_params) or None if no data available
            
            x_data: list of float - Time values in microseconds
            y_data: list of float - Signal intensity values (converted from integers)
            tof_params: dict containing:
                - ErrFlags (int): Error status bits (0=ADC underflow, 1=ADC overflow)
                        Check (ErrFlags & 1) for ADC underflow,
                              (ErrFlags & 2) for ADC overflow
                - ProtoNum (int): Protocol number used for this spectrum
                - SpecificIonCount (float): Specific ion count for this spectrum
                - SpecNum (int): REPLACED WITH num_spectra_per_trace * RecordsPerSpectrum
                - TagNum (int): Tag number for this spectrum  
                - TimeStamp (float): Timestamp in seconds
                - TotalIonCount (float): Total ion count for this spectrum
        """

        result = self._call_method('get_trace_dither')
        if result is None:
            return None
        return tuple(result)
