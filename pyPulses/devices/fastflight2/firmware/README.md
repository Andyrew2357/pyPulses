# FastFlight2 Firmware

This directory contains the FPGA bitfiles and DSP firmware required to
initialise the Signal Recovery FastFlight2 repetitive signal averager on
cold start (i.e. after power-on, before any software has communicated with
the device).

## Files

| File | Target chip |
|---|---|
| `AcqControl.rbf` | 0x4 |
| `pipes.rbf` | 0x6 |
| `pipes4P2.rbf` | 0x6 |
| `compressionfpga.rbf` | 0x8 |
| `00_AnalogFPGA.rbf` | 0xc |
| `TrigProcFPGA.rbf` | 0xc |
| `fanout.bin` | 0xa |

## Provenance

These files were obtained from the Signal Recovery product support page for
the FastFlight2.  A free account is required to access the download; the
files are otherwise publicly available at no charge.

The FastFlight2 is a discontinued product (Ametek / Signal Recovery).  No
firmware updates have been issued for many years and none are expected.  The
files are vendored here to avoid requiring users of this library to locate
and download them separately.

If Signal Recovery or Ametek wish these files to be removed, please open an
issue and they will be taken down promptly.

## Usage

Under normal circumstances these files are loaded automatically by the
driver when the device is first powered on.  Once the firmware is running it
persists until the device is power-cycled; subsequent connections within the
same power session skip the upload entirely.