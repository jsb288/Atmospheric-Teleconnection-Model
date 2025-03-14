import platform
import subprocess

import numpy as np
from torch_harmonics.quadrature import legendre_gauss_weights


def get_model_data_path(custom_path, expname):
    """Get the output datapath used for the model.

    If custom_path was set, use that as the datapath.
    Otherwise create an appropriate datapath for the user's
    operating system.
    """
    user_platform = platform.system() if (custom_path is None) else "Custom Path"
    print("Setting output datapath for", user_platform)
    datapath = ''
    match user_platform:
        case 'Custom Path':
            datapath = custom_path
        case 'Windows':
            whoami = str(subprocess.check_output(['whoami']))
            end = len(whoami) - 5
            uname = whoami[2:end].split("\\\\")[1]
            datapath = (
                "C:\\Users\\" + uname + "\\Documents\\AGCM_Experiments\\"
                + expname + "\\")
        case 'Darwin':
            whoami = str(subprocess.check_output(['whoami']))
            end = len(whoami) - 3
            uname = whoami[2:end]
            datapath = (
                '/Users/' + uname + '/Documents/AGCM_Experiments/'
                + expname + '/')
        case _:
            raise Exception(
                "Use case for this system/OS is not implemented."
                " Consider using custom_path in the advanced variables.")
    print("datapath =", datapath)

    return datapath


def precompute_latitudes(nlat, a=-1.0, b=1.0):
    """Convenience routine to precompute latitudes."""
    xlg, wlg = legendre_gauss_weights(nlat, a=a, b=b)
    lats = np.flip(np.arccos(xlg)).copy()
    wlg = np.flip(wlg).copy()

    return xlg, wlg, lats
