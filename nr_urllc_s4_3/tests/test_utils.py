import numpy as np
from nr_urllc import utils


def test_complex_exp_dtype():
    theta = np.array([0, np.pi / 2], dtype=np.float32)
    z = utils.complex_exp(theta)
    assert z.dtype == np.complex64
