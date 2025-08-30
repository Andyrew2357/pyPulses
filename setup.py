from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension

import sysconfig
import pybind11
import sys
import os
import warnings

"""
We handle any shenanigans necessary for C/C++ subroutines here, rather than 
pyproject.toml
"""

def find_file(filename, search_paths):
    """Return the first directory containing filename, or None."""
    for path in search_paths:
        candidate = os.path.join(path, filename)
        if os.path.isfile(candidate):
            return path
    return None

def get_nidaqmx_kwargs() -> dict | None:
    """
    Return extension kwargs NI-DAQmx depending on OS.
    """

    include_candidates = []
    lib_candidates = []
    try:
        if "NIDAQMX_INCLUDE" in os.environ:
                include_candidates.append(os.environ["NIDAQMX_INCLUDE"])
        if "NIDAQMX_LIB" in os.environ:
            lib_candidates.append(os.environ["NIDAQMX_LIB"])
    except:
        warnings.warn("NI-DAQmx not supported on this platform.")
        return None

    if sys.platform.startswith("win"):
        include_candidates += [
            # R"C:\Program Files (x86)\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include",
            R"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include",
        ]
        lib_candidates += [
            # R"C:\Program Files (x86)\National Instruments\NI-DAQ\DAQmx ANSI C Dev\lib\msvc",
            R"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib64\msvc",
        ]
        include_dir = find_file("NIDAQmx.h", include_candidates)
        lib_dir = find_file("NIDAQmx.lib", lib_candidates)
        libraries = ["NIDAQmx"]
        extra_compile_args = ["/std:c++17"]

    elif sys.platform.startswith("linux"):
        include_candidates += [
            "/usr/local/natinst/nidaqmx/include",
            "/usr/include/nidaqmx",
        ]
        lib_candidates += [
            "/usr/local/natinst/nidaqmx/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ]
        include_dir = find_file("NIDAQmx.h", include_candidates)
        lib_dir = find_file("libnidaqmx.so", lib_candidates)
        libraries = ["nidaqmx"]
        extra_compile_args = ["-std=c++17"]

    else:
        warnings.warn("NI-DAQmx not supported on this platform.")
        return None

    if not include_dir or not lib_dir:
        warnings.warn(
            "NI-DAQmx headers or libraries not found. "
            "The pcm1704_driver extension will not be built."
        )
        return None

    return {
        'sources'     : ["pyPulses/devices/subroutines/pcm1704_driver.cpp"],
        'include_dirs': [pybind11.get_include(), 
                         sysconfig.get_paths()["include"], 
                         include_dir,],
        'library_dirs': [lib_dir],
        'libraries'   : libraries,
        'language'    : 'c++',
        'extra_compile_args': extra_compile_args,
    }

ext_modules = []

nidaqmx_kwargs = get_nidaqmx_kwargs()
if nidaqmx_kwargs:
    ext_modules.append(
        Extension(
            "pyPulses.devices.subroutines.pcm1704_driver",
            **nidaqmx_kwargs,
        )
    )

setup(
    ext_modules = ext_modules, 
    cmdclass = {"build_ext": build_ext},
)