"""Kalman filter — process noise"""

# Fractional process noise coefficient for the complex gain A.
# Q is constructed as: Q = PROCESS_NOISE_COEFF^2 * J @ diag(|A|^2, (2π)^2) @ J^T
# where J is the Jacobian from polar to Cartesian coordinates.
# This makes noise scale with the magnitude and expected phase drift of A.
PROCESS_NOISE_COEFF = 0.01

"""
Kalman filter — initial uncertainty
"""

# Initial covariance multiplier for A after a three-point balance.
CAP_INIT_P_MULT_THREE_POINT = 1.0

# Initial covariance multiplier for A after a two-point balance.
CAP_INIT_P_MULT_TWO_POINT = 1.0