import os

visa_dll = 'C:/Windows/System32/visa64.dll'

def find_and_load_gpib(paths = [], logger = None):
    """
    Search common locations for the GPIB library and load it if found. 
    Returns True if successful, False otherwise.
    """

    # Common paths where the GPIB DLL might be located
    potential_paths = list(paths)
    potential_paths.extend([
        r'C:\Windows\System32\gpib-32.dll',
        r'C:\Windows\System32\niGPIBsys.dll',
        r'C:\Program Files\National Instruments\Shared\NI-488.2\niGPIBsys.dll',
        r'C:\Program Files (x86)\National Instruments\Shared\NI-488.2\niGPIBsys.dll'
    ])

    # Search for the DLL
    for path in potential_paths:
        if os.path.exists(path):
            try:
                import gpib_ctypes.gpib
                gpib_ctypes.gpib.gpib._load_lib(path)
                if logger:
                    logger.info(f"Successfully loaded GPIB library from {path}")
                return True
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Found GPIB library at {path} but failed to load: {e}"
                    )
                continue
    
    if logger:
        logger.error(
            "Could not find or load GPIB library in any common location."
        )
    return False