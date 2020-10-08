from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z

# dictionary for resonance masses for the different form factors
mresdict = {'A0': 0,'A1': 2,'A2': 2,'V': 1}

# resonance masses used in 2007.06957 (table XV); (pseudoscalar, vector and axial, in this order) 

mres_bc[0] = [6.275, 6.872, 7.25]
mres_bc[1] = [6.335, 6.926, 7.02, 7.28]
mres_bc[2] = [6.745, 6.75, 7.15, 7.15]

# z-parameter, see Eq.(25) of 2007.06957 
def zs_bc(q2,t0):
    zq2 = z(par['m_B'], par['m_D*+'], q2, t0)
    return np.array([1, zq2, zq2**2, zq2**3, zq2**4])

# pole function P(q^2), see Eq.(26) of 2007.06957 
def Ppole(ff,q2):
    mpole = mres_bc[mresdict[ff]]
    return np.prod([z(par['m_B'], par['m_D*+'],q2,mp**2) for mp in mpole])

process_dict = {}
process_dict['Bc->J/psi'] =  {'B': 'Bc', 'V': 'J/psi',  'q': 'b->c'}

def ffBc( q2, par, n=4):
    r"""Central value of $Bc\to J/Psi$ form factors in the lattice convention
    and simplified series expansion (SSEbc) parametrization.

    The lattice convention defines the form factors
    $A_0$, $A_1$, $A_{2}$, $V$.

    The SSEbc defines
    $$F_i(q^2) = 1/P_i(q^2) \sum_k a_k^i \,z(q^2)^k$$
    where $P_i(q^2)= \prod_{poles} z(q^2,m^2_{pole})$ is a simple pole.
    """
 
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    tm = (mB-mV)**2.
    ff = {}

    for i in ["A0","A1","A2","V"]:
        a = [ par['Bc->J/psi' + ' SSEbc ' + i.lower() + '_' + 'a' + str(j)] for j in range(n) ]
        ff[i] = Ppole(i, q2)*np.dot(a, zs(mB, mV, q2, t0=(mB-mV)**2)[:n])
    return ff
