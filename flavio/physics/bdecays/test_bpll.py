import unittest
import numpy as np
from .bpll import *


s = 1.519267515435317e+24

par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','B+'): 5.27929,
    ('mass','Bd'): 5.27961,
    ('mass','K0'): 0.497611,
    ('mass','K+'): 0.493677,
    ('lifetime','B+'): 1638.e-15*s,
    ('mass','b','MSbar'): 4.18,
    'Gmu': 1.1663787e-5,
    'alphaem': 1/127.940,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
# table XII of 1509.06235v1
    ('formfactor','B->K','a0_f+'): 0.466,
    ('formfactor','B->K','a1_f+'): -0.885,
    ('formfactor','B->K','a2_f+'): -0.213,
    ('formfactor','B->K','a0_f0'): 0.292,
    ('formfactor','B->K','a1_f0'): 0.281,
    ('formfactor','B->K','a2_f0'): 0.150,
    ('formfactor','B->K','a0_fT'): 0.460,
    ('formfactor','B->K','a1_fT'): -1.089,
    ('formfactor','B->K','a2_fT'): -1.114,
}

wc = {
    'C7': 0,
    'C7p': 0,
    'C9': 0,
    'C9p': 0,
    'C10': 0,
    'C10p': 0,
    'CP': 0,
    'CPp': 0,
    'CS': 0,
    'CSp': 0,
}

class TestBPll(unittest.TestCase):
    def test_bkll(self):
        # just some trivial tests to see if calling the functions raises an error
        prefactor(1., par, 'B+', 'K+', 'mu')
        a = amps(1., wc, par, 'B+', 'K+', 'mu')
        ac = angulardist(a, 1., par, 'B+', 'K+', 'mu')
        dGdq2(ac)
        AFB(ac)
        FH(ac)
        # rough numerical test for branching ratio at high q^2 comparing to 1510.02349
        self.assertAlmostEqual(bpll_dbrdq2(16., wc, par, 'B+', 'K+', 'mu')*1e8, 4.615/2., places=3)