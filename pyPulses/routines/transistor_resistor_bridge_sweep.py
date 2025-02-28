# from ..devices import pulseGenerator, watdScope, DeviceRegistry
# from ..utils import ExtrapPred1d, balance1d, BalanceConfig, BrentSolver 
# from ..utils import getQuickLogger, RootFinderStatus
# import numpy as np
# import argparse

# def main(args):
#     debug_folder    = args.debug
#     gate_channel    = args.Vgchannel
#     y_tolerance     = args.ytolerance
#     x_tolerance     = args.xtolerance
#     max_iter        = args.iterations
#     bias_points     = np.linspace(args.Vgstart, args.Vgend, args.Vgsteps)

#     watdScope_logger = getQuickLogger("watdScope", debug_folder)
#     mso44_logger = getQuickLogger("mso44", debug_folder)

#     watd_config = {

#     }

#     watd = watdScope(
#         loggers     = (watdScope_logger, mso44_logger),
#         config      = watd_config    
#     )
#     watd.scope.set_acquisition_mode("AVER")
#     watd.scope.set_num_averages(10e3)
#     watd.scope.fast_acquisition(True)

#     # Set-up pulse generator logging
#     pulseGenerator_logger = getQuickLogger("pulseGenerator", debug_folder)
#     ad5764_logger = getQuickLogger("ad5764", debug_folder)
#     dtg5274_logger = getQuickLogger("dtg5274", debug_folder)

#     pulse_gen = pulseGenerator(
#         loggers = (pulseGenerator_logger, ad5764_logger, dtg5274_logger),
#     )
#     pulse_gen.exc_on(True)
#     pulse_gen.dis_on(False)
#     pulse_gen.set_prate("SIXT")
#     pulse_gen.set_polarity(True)
#     pulse_gen.set("Vx1", 2.0)
#     pulse_gen.dtg.set_frequency(10e6)
#     dc_box = pulse_gen.dcbox

#     predictor = ExtrapPred1d(
#         support     = 5,
#         order       = 3,
#         default0    = lambda x: 0.5*x,
#         default1    = lambda x, *args: 2*x
#     )

#     balance1d_logger = getQuickLogger("balance1d")

#     balance_config = BalanceConfig(
#         set_x           = lambda x: pulse_gen.set_V("Vy1", x),
#         get_y           = lambda : watd.take_integral(),
#         predictor       = predictor,
#         rootfinder      = BrentSolver,
#         y_tolerance     = y_tolerance,
#         x_tolerance     = x_tolerance,
#         search_range    = (0, 4.3),
#         max_iter        = max_iter,
#         max_step        = None,
#         logger          = balance1d_logger
#     )

#     for Vg in bias_points:
#         dc_box.set_V(gate_channel, Vg)

#         balance_status = balance1d(Vg, balance_config)
#         Vy = pulse_gen.get_V("Vy1")
        
#         if balance_status == RootFinderStatus.CONVERGED:
#             print(f"Successfully balanced with Vy = {Vy}.")
#             predictor.update(Vg, Vy)
#         else:
#             print("Failed to find a sufficient balance point.")
#             print(f"Ended with Vy = {Vy}.")
#         print("="*40)
    
#     DeviceRegistry.clear_registry()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         prog        = "Transistor-Resistor Bridge Sweep",
#         description = """
#                         This is intended as a test of the balancing procedure. 
#                         It sweeps over gate voltage, balancing the bridge at 
#                         each bias. It uses a Brent-Dekker method for root 
#                         finding and predicts future balance points using 
#                         polynomial extrapolation.
#                       """,
#         epilog      = """

#                       """
#     )

#     parser.add_argument("-d", "--debug", type = str, required = True,
#                         help = "Path to folder for debug logs.")
#     parser.add_argument("-Vgch", "--Vgchannel", type = int, required = True,
#                         help = "DC box channel that sets the gate voltage.")
#     parser.add_argument("-ytol", "--ytolerance", type = float, required = True,
#                         help = """
#                         Allowed deviation from 0 for an integrated trace when on 
#                         balance.
#                         """)
#     parser.add_argument("-xtol", "--xtolerance", type = float, required = True,
#                         help = """
#                         Allowed uncertainty in the optimal balance point.
#                         """)
#     parser.add_argument("-iter", "--iterations", type = int, required = True,
#                         help = "Maximum root finder iterations allowed")
#     parser.add_argument("-Vgi", "--Vgstart", type = float, required = True,
#                         help = "Start of the gate bias points.")
#     parser.add_argument("-Vgf", "--Vgend", type = float, required = True,
#                         help = "End of the gate bias points.")
#     parser.add_argument("-Vgst", "--Vgsteps", type = int, required = True,
#                         help = "Number of distinct gate bias points to take.")

#     main(parser.parse_args())
