# pyPulses
Instrument control code for pulsed electronic measurements, intended for use in an IPython notebook.

## Quick Installation Instructions
1. Clone the repository: <code>git clone https://github.com/Andyrew2357/pyPulses.git</code>
2. Create a virtual environment in your intended operating directory and activate that environment.
3. Navigate to the cloned repository and run: <code>python -m pip install -e .</code>. The <code>-e</code> flag will make this an editable install. Therefore any changes to the cloned repository will be reflected upon importing.
4. To use the package, create a jupyter notebook and import from <code>pyPulses</code>, <code>pyPulses.devices</code>, <code>pyPulses.utils</code>, etc.
5. If you are working in Visual Studio Code, it helps to edit your <code>settings.json</code> to include <code>"python.analysis.extraPaths": ["path/to/cloned/repository"]</code> and <code>"python.languageServer": "Pylance"</code>. This will ensure proper code highlighting and IntelliSense.

## Linux Setup

Some instruments use raw USB (bypassing NI-DAQmx entirely) and others use VISA. Both require some system-level setup on Linux that cannot be handled by pip.

### USB device permissions (all USB instruments)

By default, Linux restricts access to USB devices to root. The fix is a udev rule that grants world read/write access when a recognised device is plugged in. pyPulses ships the relevant rules. Copy them to `/etc/udev/rules.d/` and reload:

```bash
sudo cp path/to/pyPulses/pyPulses/devices/udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Replug any connected USB instruments after reloading. You can verify a rule applied correctly with:

```bash
ls -l /dev/bus/usb/<BUS>/<DEV>   # should show crw-rw-rw-
```

where `<BUS>` and `<DEV>` come from `lsusb` output.

**NI USB-6501 only:** the `ni_usb6501` kernel module (part of the comedi framework, not NI-DAQmx) is loaded automatically on plug-in and will claim the device before pyPulses can open it. Blacklist it:

```bash
echo "blacklist ni_usb6501" | sudo tee /etc/modprobe.d/blacklist-ni-usb6501.conf
sudo modprobe -r ni_usb6501
```

This only needs to be done once.

### VISA instruments (GPIB, TCPIP, Serial)

pyPulses uses [pyvisa-py](https://pyvisa.readthedocs.io/projects/pyvisa-py/en/stable/) as its VISA backend, which is a pure-Python implementation that does not require NI-VISA. It is installed automatically as a dependency.

For TCPIP and Serial instruments this is sufficient. For GPIB you also need the `linux-gpib` kernel driver and userspace library (see below).

### GPIB support

GPIB requires a kernel module and userspace library that must be built from source. First install the build dependencies:

```bash
sudo apt install build-essential subversion linux-headers-$(uname -r) \
    autoconf automake libtool flex bison python3-dev
```

The official linux-gpib project is hosted on Sourceforge SVN, but the SVN trunk may not build against recent kernels (6.7+) due to kernel API changes. If your kernel is 6.7 or newer, use the community-maintained GitHub mirror which tracks these changes:

```bash
# Prefer this on kernel 6.7+
git clone https://github.com/coolshou/linux-gpib.git

# Otherwise use the official SVN trunk
# svn checkout svn://svn.code.sf.net/p/linux-gpib/code/trunk linux-gpib
```

Build and install the kernel module:

```bash
cd linux-gpib/linux-gpib-kernel
make && sudo make install
```

Build and install the userspace library. The `--sysconfdir=/etc` flag is important on Ubuntu — without it the udev rules end up in `/usr/local/etc` where udev won't find them:

```bash
cd ../linux-gpib-user
./bootstrap && ./configure --sysconfdir=/etc
make && sudo make install
sudo ldconfig
```

Install the Python bindings into your virtual environment. Do not use `sudo python3` — that would install into the system Python, not your environment. Call your virtualenv's interpreter directly:

```bash
cd language/python
/path/to/your/venv/bin/python setup.py install
```

**Verify the Python bindings installed correctly** (adapter does not need to be plugged in for this):

```bash
/path/to/your/venv/bin/python -c "import Gpib; print('ok')"
```

Note that the linux-gpib bindings are imported as `Gpib` (capital G), not `gpib` or `visa`. Also confirm that pyvisa-py itself is present in your environment:

```bash
/path/to/your/venv/bin/pip show pyvisa-py
```

**Fix `/etc/gpib.conf`:** the default config shipped by linux-gpib specifies `board_type = "ni_pci"`, which is wrong for a USB adapter. Edit it:

```bash
sudo sed -i 's/board_type = "ni_pci"/board_type = "ni_usb_b"/' /etc/gpib.conf
```

Substitute `ni_usb_b` with the appropriate type for your adapter if you are not using the NI GPIB-USB-HS.

**Set permissions on `/dev/gpib0`:** linux-gpib creates the gpib device node when the adapter is plugged in, but by default it is only accessible by root. The udev rules installed by linux-gpib use `GROUP="gpib"`, but this can be unreliable on Ubuntu due to GID resolution issues at module load time. The simplest fix for a single-user lab machine is to make the node world-readable:

```bash
cat << 'EOF' | sudo tee /etc/udev/rules.d/98-gpib-generic.rules
KERNEL=="gpib[0-9]*", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

If `/dev/gpib0` already exists, the trigger alone may not update it. In that case reload the module (with the adapter plugged in):

```bash
sudo modprobe -r ni_usb_gpib gpib_common
sudo modprobe gpib_common
sudo modprobe ni_usb_gpib
```

Then verify:

```bash
ls -l /dev/gpib0    # should show crw-rw-rw-
```

**Verify the kernel module loads correctly** (plug in the GPIB adapter first):

```bash
lsmod | grep gpib              # should show gpib_common and ni_usb_gpib
```

Note that `lsmod | grep gpib` will return nothing if the adapter is not plugged in — the USB module only loads when the hardware is present.

**Verify end-to-end with `ibtest`:** with the adapter plugged in and an instrument connected and powered on, run:

```bash
ibtest
```

Choose `b` for board, then enter `violet` (the default board name in `gpib.conf`) when prompted. If the board opens successfully, choose `d` for device and enter your instrument's GPIB address to test communication. `ENOL: No listeners` at this stage means the software stack is working but the instrument is not responding — check the GPIB cable, that the instrument is powered on, and that the address matches what is set on the instrument's front panel.

Finally, with the adapter plugged in and the virtualenv active, confirm that pyvisa-py sees GPIB as available:

```bash
python -m visa info
```

Look for `GPIB INSTR: Available via Linux GPIB` in the output. If it instead reports that the library could not be located, ensure `sudo ldconfig` was run after installing the userspace library.
