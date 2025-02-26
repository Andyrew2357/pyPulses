from ..devices import pulseGenerator, watdScope, DeviceRegistry
from ..utils import getQuickLogger, tandemSweep
import numpy as np
# import argparse

def main(args):
    max_height      = args.maxh
    nsteps          = args.nstep
    x_path          = args.xpath
    y_path          = args.ypath
    s_path          = args.spath
    debug_folder    = args.debug
    dcbox_map       = args.dcbox_map
    wait = 0.05

    print(f"max_height = {max_height}")
    print(f"nsteps = {nsteps}")
    print(f"x_path = {x_path}")
    print(f"y_path = {y_path}")
    print(f"s_path = {s_path}")
    print(f"debug_folder = {debug_folder}")

    dcbox_logger = getQuickLogger("ad5764", debug_folder)
    dtg_logger = getQuickLogger("dtg5274", debug_folder)
    mso44_logger = getQuickLogger("mso44", debug_folder)
    pGen_logger = getQuickLogger("pulseGenerator", debug_folder)
    watd_logger = getQuickLogger("watdScope", debug_folder)

    print("Loading pulse generator...")
    pulse_gen = pulseGenerator(
        loggers = (pGen_logger, dcbox_logger, dtg_logger)
    )
    pulse_gen.max_V = max_height
    pulse_gen.dcbox_map = dcbox_map

    print("Loading watd scope...")
    watd = watdScope(loggers = (watd_logger, mso44_logger))

    print("Sweeping pulses and counterpulses in tandem...")
    with open(s_path, 'w') as sfile:
        sfile.write("V, SUM, *TRACE\n")
        for V in np.linspace(0, max_height, nsteps):
            print(f"Vx = Vy = {V}")
            tandemSweep(wait, 
                (lambda x: pulse_gen.set_V("Vx1", x), pulse_gen.get_V("Vx1"), V, 0.1),
                (lambda x: pulse_gen.set_V("Vy1", x), pulse_gen.get_V("Vy1"), V, 0.1)
            )
            pulse_gen.set_V("Vx1", V)
            pulse_gen.set_V("Vy1", V)
            watd.scope.set_channel(1)
            a = watd.take_integral()
            va = watd.get_waveform()[1]
            watd.scope.set_channel(2)
            b = watd.take_integral()
            vb = watd.get_waveform()[1]
            wave = ''.join([f"{v}, " for v in (va + vb)])[:-1]
            sfile.write(f"{V}, {a + b}, {wave}\n")

    print("Sweeping pulses and counterpulses to zero...")
    tandemSweep(wait, 
        (lambda x: pulse_gen.set_V("Vx1", x), pulse_gen.get_V("Vx1"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vy1", x), pulse_gen.get_V("Vy1"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vx2", x), pulse_gen.get_V("Vx2"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vy2", x), pulse_gen.get_V("Vy2"), 0., 0.1)
    )

    print("Sweeping pulses...")
    watd.scope.set_channel(1)
    with open(x_path, 'w') as xfile:
        xfile.write("Vx, OUT, *TRACE\n")
        for Vx in np.linspace(0, max_height, nsteps):
            print(f"Vx = {Vx}")
            pulse_gen.set_V("Vx1", Vx, 0.1, wait)
            integral = watd.take_integral()
            wave = ''.join([f"{v}, " for v in watd.get_waveform()[1]])[:-1]
            xfile.write(f"{Vx}, {integral}, {wave}\n")
    
    pulse_gen.set_V("Vx1", 0., 0.1, wait)

    print("Sweeping counterpulses...")
    watd.scope.set_channel(2)
    with open(y_path, 'w') as yfile:
        yfile.write("Vy, OUT, *TRACE\n")
        for Vy in np.linspace(0, max_height, nsteps):
            print(f"Vy = {Vy}")
            pulse_gen.set_V("Vy1", Vy, 0.1, wait)
            integral = watd.take_integral()
            wave = ''.join([f"{v}, " for v in watd.get_waveform()[1]])[:-1]
            yfile.write(f"{Vy}, {integral}, {wave}\n")
    
    print("Sweeping pulses and counterpulses to zero...")
    tandemSweep(wait, 
        (lambda x: pulse_gen.set_V("Vx1", x), pulse_gen.get_V("Vx1"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vy1", x), pulse_gen.get_V("Vy1"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vx2", x), pulse_gen.get_V("Vx2"), 0., 0.1),
        (lambda x: pulse_gen.set_V("Vy2", x), pulse_gen.get_V("Vy2"), 0., 0.1)
    )

    DeviceRegistry.clear_registry()
