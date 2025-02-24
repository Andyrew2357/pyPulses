from ..devices import pulseGenerator, watdScope, DeviceRegistry
from ..utils import getQuickLogger, tandemSweep
import numpy as np
import argparse

def main(args):
    max_height      = args.maxh
    nsteps          = args.nstep
    x_path          = args.xpath
    y_path          = args.ypath
    s_path          = args.spath
    debug_folder    = args.debug

    dcbox_logger = getQuickLogger("ad5764", debug_folder)
    dtg_logger = getQuickLogger("dtg5274", debug_folder)
    mso44_logger = getQuickLogger("mso44", debug_folder)
    pGen_logger = getQuickLogger("pulseGenerator", debug_folder)
    watd_logger = getQuickLogger("watdScope", debug_folder)

    pulse_gen = pulseGenerator(
        loggers = (pGen_logger, dcbox_logger, dtg_logger)
    )
    pulse_gen.max_V = max_height

    watd = watdScope(loggers = (watd_logger, mso44_logger))

    tandemSweep(0.05, 
        (lambda x: pulse_gen.set_V("Vx1", x), pulse_gen.get_V("Vx1"), 0., 0.05),
        (lambda x: pulse_gen.set_V("Vy1", x), pulse_gen.get_V("Vy1"), 0., 0.05),
        (lambda x: pulse_gen.set_V("Vx2", x), pulse_gen.get_V("Vx2"), 0., 0.05),
        (lambda x: pulse_gen.set_V("Vy2", x), pulse_gen.get_V("Vy2"), 0., 0.05)
    )

    with open(x_path, 'w') as xfile:
        xfile.write("Vx    OUT\n")
        for Vx in np.linspace(0, max_height, nsteps):
            pulse_gen.sweep_V("Vx1", Vx, 0.05, 0.05)
            xfile.write(f"{Vx}    {watd.take_integral()}")
    
    pulse_gen.sweep_V("Vx1", 0., 0.05, 0.05)

    with open(y_path, 'w') as yfile:
        yfile.write("Vy    OUT\n")
        for Vy in np.linspace(0, max_height, nsteps):
            pulse_gen.sweep_V("Vy1", Vy, 0.05, 0.05)
            yfile.write(f"{Vy}    {watd.take_integral()}")

    with open(s_path, 'w') as sfile:
        sfile.write("V    OUT\n")
        for V in np.linspace(0, max_height, nsteps):
            pulse_gen.set_V("Vx1", V)
            pulse_gen.set_V("Vy1", V)
            sfile.write(f"{V}    {watd.take_integral()}")

    DeviceRegistry.clear_registry()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog        = "Benchmark Pulse Generator",
        description = """
                        This is intended as a test of the amplitude range of the
                        pulse generator. It sweeps both the pulse and
                        counterpulse and takes another sweep of the two together.
                      """,
        epilog      = """

                      """
    )

    parser.add_argument("-d", "--debug", type = str, required = True,
                        help = "Path to folder for debug logs.")
    parser.add_argument("-x", "--xpath", type = str, required = True,
                        help = "Path to pulse output")
    parser.add_argument("-y", "--ypath", type = str, required = True,
                        help = "Path to counterpulse output")
    parser.add_argument("-s", "--spath", type = str, required = True,
                        help = "Path to summed pulses output")
    parser.add_argument("-n", "--nstep", type = int, required = True,
                        help = "Number of steps to take in a sweep")
    parser.add_argument("-m", "--maxh", type = float, required = True,
                        help = "Maximum pulse height to apply.")

    main(parser.parse_args())
