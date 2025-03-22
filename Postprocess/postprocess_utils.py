import os

import numpy as np
from torch_harmonics.quadrature import legendre_gauss_weights


def get_model_data_path(custom_path, expname):
    """Get the output datapath used for the model.

    If custom_path was set, use that as the datapath.
    Otherwise create an appropriate datapath for the user's
    operating system.
    """
    path_type = "Documents Folder" if (custom_path is None) else "Custom Path"
    print("Setting output datapath to", path_type)
    datapath = ''
    if custom_path is None:
        datapath = os.path.join(
            "~", "Documents", "AGCM_Experiments", expname, "")
        datapath = os.path.expanduser(datapath)
    else:
        datapath = custom_path
    print("datapath =", datapath)

    return datapath


def precompute_latitudes(nlat, a=-1.0, b=1.0):
    """Convenience routine to precompute latitudes."""
    xlg, wlg = legendre_gauss_weights(nlat, a=a, b=b)
    lats = np.flip(np.arccos(xlg)).copy()
    wlg = np.flip(wlg).copy()

    return xlg, wlg, lats
