from .ff_package_manager import FastFlightPackageManager

import json
import subprocess
import time
import struct
import numpy as np
from typing import Tuple

class FastFlight64():
    def __init__(self):
        self.package_manager = FastFlightPackageManager()
        self.python32_path = self.package_manager.get_python32_path()
        self.bridge_script_path = self.package_manager.get_bridge_script_path()

        self._process = None
        self._start_bridge()

    def _start_bridge(self):
        """Start the 32-bit bridge process"""

        if self._process is not None:
            return
        
        try:
            self._process = subprocess.Popen(
                [self.python32_path, self.bridge_script_path],
                stdin   = subprocess.PIPE,
                stdout  = subprocess.PIPE,
                stderr  = subprocess.PIPE,
                text    = False,
                bufsize = 0
            )

            time.sleep(0.5)

            # Check if the process started successfully
            if self._process.poll() is not None:
                # Process already terminated
                stderr_output = self._process.stderr.read().decode('utf-8')
                stdout_output = self._process.stdout.read().decode('utf-8')
                raise RuntimeError(
                    f"Bridge process terminated immediately."
                    f"STDERR: {stderr_output}, STDOUT: {stdout_output}"
                )

            print("Bridge process started, attempting to initialize...")

            self._call_method('init')

        except Exception as e:
            raise RuntimeError(f"Failed to start 32-bit Python bridge: {e}")
        
    def _reconstruct_x_data(self, length: int, sampling_interval: int
                            ) -> np.ndarray:
        """Reconstruct X data from sampling interval and length"""

        interval_map = {
            0: 0.25,
            1: 0.25,
            2: 0.5,
            3: 1.0,
            4: 2.0
        }

        return interval_map[sampling_interval] * np.arange(length)
    
    def _read_exact(self, size: int) -> bytes:
        """Read exactly 'size' bytes from stdout, handling partial reads"""
        data = b''
        while len(data) < size:
            chunk = self._process.stdout.read(size - len(data))
            if not chunk:
                raise RuntimeError(
                    f"Unexpected end of stream. Expected {size} bytes, "
                    f"got {len(data)}")
            data += chunk
        return data
    
    def _read_binary_data(self):
        """Read binary data response from bridge process"""
        try:
            # Read fixed-size metadata: sampling_interval(4) + err_flags(4) + 
            # proto_num(4) + spec_num(4) + timestamp(8) = 24 bytes
            metadata_bytes = self._read_exact(24)
            
            sampling_interval, err_flags, proto_num, spec_num = \
                struct.unpack('<IIII', metadata_bytes[:16])
            timestamp = struct.unpack('<d', metadata_bytes[16:24])[0]
            
            # Read data length
            data_len_bytes = self._read_exact(4)
            data_length = struct.unpack('<I', data_len_bytes)[0]
            
            if data_length == 0:
                # Handle empty data case
                x_data = np.array([])
                y_data = np.array([])
            else:
                # Read Y data
                y_bytes = self._read_exact(data_length * 4)  # 4 bytes per int32
                y_data = np.frombuffer(y_bytes, dtype='<i4')  # Little-endian int32
                x_data = self._reconstruct_x_data(data_length, sampling_interval)
            
            tof_parms = {
                'ErrFlags'  : err_flags,
                'ProtoNum'  : proto_num,
                'SpecNum'   : spec_num,
                'TimeStamp' : timestamp
            }
            
            return x_data, y_data.astype(float), tof_parms
            
        except Exception as e:
            raise RuntimeError(f"Failed to read binary data: {e}")
        
    def _call_method(self, method, *args, **kwargs):
        """Call a method on the remote FastFlight instance"""

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Bridge process is not running")
        
        request = {
            'method': method,
            'args'  : list(args),
            'kwargs': kwargs
        }

        try:
            # send request
            request_json = json.dumps(request) + '\n'
            self._process.stdin.write(request_json.encode('utf-8'))
            self._process.stdin.flush()

            # Special handling for get_data which returns binary
            if method in ['get_data', 'get_trace', 'get_trace_dither']:
                # Read the first line to see if it's a JSON or binary
                response_line_bytes = self._process.stdout.readline()
                if not response_line_bytes:
                    raise RuntimeError("No response from bridge process")
                
                response_line = response_line_bytes.decode('utf-8').strip()
                
                # Check if it's the binary data marker
                if response_line == "BINARY_DATA_FOLLOWS":
                    return self._read_binary_data()
                else:
                    # It's a regular JSON response (probably None case)
                    try:
                        response = json.loads(response_line)
                        if response['success']:
                            return response['result']
                        else:
                            error_msg = response['error']
                            if 'traceback' in response:
                                error_msg += '\n' + response['traceback']
                            raise RuntimeError(f"Remote error: {error_msg}")
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"Failed to decode JSON response: {response_line}, "
                            f"error: {e}")
                            
            else:
                # Read regular JSON response
                response_line_bytes = self._process.stdout.readline()
                if not response_line_bytes:
                    raise RuntimeError("No response from bridge process")
                
                response_line = response_line_bytes.decode('utf-8').strip()
                
                try:
                    response = json.loads(response_line)
                    if response['success']:
                        return response['result']
                    else:
                        error_msg = response['error']
                        if 'traceback' in response:
                            error_msg += '\n' + response['traceback']
                        raise RuntimeError(f"Remote error: {error_msg}")
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Failed to decode JSON response: "
                                       f"{response_line}, error: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")
        
    def __del__(self):
        """Clean up the bridge process"""
        self.close_bridge()

    def close_bridge(self):
        """Explicitly close the bridge process"""
        if self._process is None:
            return
        
        try:
            self._process.stdin.write(b'QUIT\n')
            self._process.stdin.flush()
            self._process.wait(timeout = 5)

        except:
            self._process.terminate()
            try:
                self._process.wait(timeout = 2)
            except:
                self._process.kill()

        finally:
            self._process = None

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
