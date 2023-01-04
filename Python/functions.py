import numpy as np
import scipy.constants as c

def permitivity_drude(omega, omega_p, gamma, epsilon_0=1):
    return epsilon_0 * ( 1 - omega_p**2 / (omega**2 + 1j * gamma * omega) )