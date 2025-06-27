from .ff_package_manager import FastFlightPackageManager

import json
import subprocess
import time
from typing import Optional, Tuple

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
                text    = True,
                bufsize = 0
            )

            time.sleep(0.5)

            # Check if the process started successfully
            if self._process.poll() is not None:
                # Process already terminated
                stderr_output = self._process.stderr.read()
                stdout_output = self._process.stdout.read()
                raise RuntimeError(
                    f"Bridge process terminated immediately."
                    f"STDERR: {stderr_output}, STDOUT: {stdout_output}"
                )

            print("Bridge process started, attempting to initialize...")

            self._call_method('init')

        except Exception as e:
            raise RuntimeError(f"Failed to start 32-bit Python bridge: {e}")
        
    def _call_method(self, method, *args, **kwargs):
        """Call a method on the remote FastFlight instance"""

        if self._process is None or \
            self._process.poll() is not None:
            raise RuntimeError("Bridge process is not running")
        
        request = {
            'method': method,
            'args'  : list(args),
            'kwargs': kwargs
        }

        try:
            # send request
            self._process.stdin.write(json.dumps(request) + '\n')
            self._process.stdin.flush()

            # read response
            response_line = self._process.stdout.readline().strip()
            if not response_line:
                raise RuntimeError("No response from bridge process")
            
            response = json.loads(response_line)

            if response['success']:
                return response['result']
            else:
                error_msg = response['error']
                if 'traceback' in response:
                    error_msg += '\n' + response['traceback']
                raise RuntimeError(f"Remote error: {error_msg}")
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode response: {e}")
        
    def __del__(self):
        """Clean up the brige process"""
        self.close_bridge()

    def close_bridge(self):
        """Explicitly close the bridge process"""
        if self._process is None:
            return
        
        try:
            self._process.stdin.write('QUIT\n')
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
    
    def is_acq_running(self) -> Optional[bool]:
        return self._call_method('is_acq_running')
    
    def device_count(self) -> Optional[int]:
        return self._call_method('device_count')
    
    def FF_version(self) -> Optional[str]:
        return self._call_method('FF_version')
    
    def is_connected(self) -> Optional[bool]:
        return self._call_method('is_connected')
    
    def num_records(self) -> Optional[int]:
        return self._call_method('num_records')
    
    def serial_number(self) -> Optional[str]:
        return self._call_method('serial_number')
    
    def num_spectra(self) -> Optional[int]:
        return self._call_method('num_spectra')
    
    def time_elapsed(self) -> Optional[float]:
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
    
    def get_data(self) -> Optional[Tuple[list, list, dict]]:
        result = self._call_method('get_data')
        if result is None:
            return None
        return tuple(result)
    