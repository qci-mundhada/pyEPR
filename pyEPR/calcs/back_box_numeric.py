'''
Numerical diagonalization of quantum Hamiltonian and parameter
extraction.

@author: Phil Reinhold, Zlatko Minev, Lysander Christakis

Original code on black_box_hamiltonian and make_dispersive functions by Phil Reinhold
Revisions and updates by Zlatko Minev & Lysander Christakis
'''
# pylint: disable=invalid-name


from __future__ import print_function

from functools import reduce

import numpy as np
import time
import itertools as it
from .constants import Planck as h
from .constants import fluxQ, hbar
from .hamiltonian import MatrixOps

try:
    import qutip
    from qutip import basis, tensor
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [ 'epr_numerical_diagonalization',
            'make_dispersive',
            'black_box_hamiltonian',
            'black_box_hamiltonian_nq']

dot = MatrixOps.dot
cos_approx = MatrixOps.cos_approx


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def epr_numerical_diagonalization(freqs, Ljs, ϕzpf,
             cos_trunc=8,
             fock_trunc=9,
             use_1st_order=None,
             return_H=False,
             sparse=False,
             all_eig = True):
    '''
    Numerical diagonalizaiton for pyEPR. Ask Zlatko for details.

    :param fs: (GHz, not radians) Linearized model, H_lin, normal mode frequencies in Hz, length M
    :param ljs: (Henries) junction linerized inductances in Henries, length J
    :param fzpfs: (reduced) Reduced Zero-point fluctutation of the junction fluxes for each mode
                across each junction, shape MxJ

    :return: Hamiltonian mode freq and dispersive shifts. Shifts are in MHz.
             Shifts have flipped sign so that down shift is positive.
    '''

    freqs, Ljs, ϕzpf = map(np.array, (freqs, Ljs, ϕzpf))
    assert(all(freqs < 1E6)
           ), "Please input the frequencies in GHz. \N{nauseated face}"
    assert(all(Ljs < 1E-3)
           ), "Please input the inductances in Henries. \N{nauseated face}"

    print("Starting Hamiltonian generation")
    start = time.time()
    Hs = black_box_hamiltonian(freqs * 1E9, Ljs.astype(np.float), fluxQ*ϕzpf,
                 cos_trunc, fock_trunc, individual=use_1st_order)
    end = time.time()
    print(f'Hamiltonain generations finished in {end-start} seconds')
    

    f_ND, χ_ND, _, _ = make_dispersive(Hs, fock_trunc, ϕzpf, freqs, use_1st_order=use_1st_order, sparse=sparse, all_eig=all_eig)

    χ_ND = -1*χ_ND * 1E-6  # convert to MHz, and flip sign so that down shift is positive

    return (f_ND, χ_ND, Hs) if return_H else (f_ND, χ_ND)




def black_box_hamiltonian(fs, ljs, fzpfs, cos_trunc=5, fock_trunc=8, individual=False):
    r"""
    :param fs: Linearized model, H_lin, normal mode frequencies in Hz, length N
    :param ljs: junction linerized inductances in Henries, length M
    :param fzpfs: Zero-point fluctutation of the junction fluxes for each mode across each junction,
                 shape MxJ
    :return: Hamiltonian in units of Hz (i.e H / h)
    All in SI units. The ZPF fed in are the generalized, not reduced, flux.

    Description:
     Takes the linear mode frequencies, $\omega_m$, and the zero-point fluctuations, ZPFs, and
     builds the Hamiltonian matrix of $H_full$, assuming cos potential.
    """
    n_modes = len(fs)
    njuncs = len(ljs)
    fs, ljs, fzpfs = map(np.array, (fs, ljs, fzpfs))
    ejs = fluxQ**2 / ljs
    fjs = ejs / h

    fzpfs = np.transpose(fzpfs)  # Take from MxJ  to JxM

    assert np.isnan(fzpfs).any(
    ) == False, "Phi ZPF has NAN, this is NOT allowed! Fix me. \n%s" % fzpfs
    assert np.isnan(ljs).any(
    ) == False, "Ljs has NAN, this is NOT allowed! Fix me."
    assert np.isnan(
        fs).any() == False, "freqs has NAN, this is NOT allowed! Fix me."
    assert fzpfs.shape == (njuncs, n_modes), "incorrect shape for zpf array, {} not {}".format(
        fzpfs.shape, (njuncs, n_modes))
    assert fs.shape == (n_modes,), "incorrect number of mode frequencies"
    assert ejs.shape == (njuncs,), "incorrect number of qubit frequencies"

    def tensor_out(op, loc):
        "Make operator <op> tensored with identities at locations other than <loc>"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)

    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)
    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    mode_ns = [tensor_out(n, i) for i in range(n_modes)]

    def cos(x):
        return cos_approx(x, cos_trunc=cos_trunc)
    
    print('fzpf shape is:',fzpfs.shape)

    linear_part = dot(fs, mode_ns)
    cos_interiors = [dot(fzpf_row/fluxQ, mode_fields) for fzpf_row in fzpfs]
    nonlinear_part = dot(-fjs, map(cos, cos_interiors))
    if individual:
        return linear_part, nonlinear_part
    else:
        return linear_part + nonlinear_part

bbq_hmt = black_box_hamiltonian

def make_dispersive(H, fock_trunc, fzpfs=None, f0s=None, chi_prime=False,
                    use_1st_order=None, sparse=False, all_eig=True):
    r"""
    Input: Hamiltonian Matrix.
        Optional: phi_zpfs and normal mode frequncies, f0s.
        use_1st_order : deprecated
    Output:
        Return dressed mode frequencies, chis, chi prime, phi_zpf flux (not reduced), and linear frequencies
    Description:
        Takes the Hamiltonian matrix `H` from bbq_hmt. It them finds the eigenvalues/eigenvectors and  assigns quantum numbers to them --- i.e., mode excitations,  such as, for instance, for three mode, |0,0,0> or |0,0,1>, which correspond to no excitations in any of the modes or one excitation in the 3rd mode, resp.    The assignment is performed based on the maximum overlap between the eigenvectors of H_full and H_lin.   If this crude explanation is confusing, let me know, I will write a more detailed one :slightly_smiling_face:
        Based on the assignment of the excitations, the function returns the dressed mode frequencies $\omega_m^\prime$, and the cross-Kerr matrix (including anharmonicities) extracted from the numerical diagonalization, as well as from 1st order perturbation theory.
        Note, the diagonal of the CHI matrix is directly the anharmonicity term.
    """

    if use_1st_order is None:
        use_1st_order = {'return_2O_PT': False, 'PT_fock_cutoff':0, 'identify_vectors_using_1O_PT':False}
    else:
        if use_1st_order.get('return_2O_PT', False):
            use_1st_order['identify_vectors_using_1O_PT'] = True

    if hasattr(H, '__len__'):  # is it an array / list?
        [H_lin, H_nl] = H
        H = H_lin + H_nl
    else:  # make sure its a quanutm object
        assert type(
            H) == qutip.qobj.Qobj, "Please pass in either  a list of Qobjs or Qobj for the Hamiltonian"


    N = int(np.log(H.shape[0]) / np.log(fock_trunc))    # number of modes
    assert H.shape[0] == fock_trunc ** N

    if sparse:
        print("Will use sparse matrix diagonalization")

    if all_eig:
        N_eigs = 0
    else:
        ground_states = 1
        one_photon_states = N
        two_photon_states = N+N*(N-1)/2
        three_photon_states = N+N*(N-1)/2*2+N*(N-1)*(N-2)/6 
        N_eigs = int(ground_states+one_photon_states+two_photon_states+three_photon_states)
        print(f"User forbids calculating all eigenvalues. Will only calculate {N_eigs} eigenvalues")
        # if use_1st_order:
        #     raise NotImplementedError("Can't use use_1st_order=True when calculating partial eigenvalues.")
        if chi_prime:
            raise NotImplementedError("Not calculating enough eigenstates to calculate chi_prime yet")
        

    if not use_1st_order['return_2O_PT']:
        print("Starting the diagonalization")
        start = time.time()
        evals, evecs = H.eigenstates(sparse=sparse,eigvals=N_eigs)
        end = time.time()
        print(f"Finished the diagonalization in {end-start} seconds")
        # evals -= evals[0]
    else:
        print('Going to return 2nd order PT results so not performing the numerical diagonalization')

    def fock_state_on(d):
        ''' d={mode number: # of photons} '''
        return qutip.tensor(*[qutip.basis(fock_trunc, d.get(i, 0)) for i in range(N)])  # give me the value d[i]  or 0 if d[i] does not exist

    if use_1st_order['identify_vectors_using_1O_PT']:
        num_modes = N
        print("Using 1st O")

        if use_1st_order['PT_fock_cutoff']:
            PT_fock_cutoff = use_1st_order['PT_fock_cutoff']
        else:
            PT_fock_cutoff = fock_trunc

        def find_multi_indices():
            multi_indices = [{ind: item for ind, item in enumerate(combo)} for combo in it.product(range(PT_fock_cutoff),repeat=N)]
            return multi_indices
            '''this function generates all possible multi-indices for three modes for a given fock_trunc'''

        def get_expect_number(left, middle, right):
            return (left.dag()*middle*right).data.toarray()[0, 0]
            '''this function calculates the expectation value of an operator called "middle" '''

        def get_basis0(remove=None):
            multi_indices = find_multi_indices()
            # print(multi_indices)
            if remove is not None:
                multi_indices.remove(remove)
            basis0 = [fock_state_on(multi_indices[i]) for i in range(len(multi_indices))]
            evalues0 = [get_expect_number(v0, H_lin, v0) for v0 in basis0]
            return multi_indices, basis0, evalues0
            '''this function creates a basis of fock states and their corresponding eigenvalues'''

        def closest_state_to(d):

            def PT_on_vector(original_vector, original_basis, energy0, evalue):
                new_vector = 0 * original_vector
                for i in range(len(original_basis)):
                    if 1: #np.abs(energy0[i]-evalue) > 1e-6:
                        new_vector += ((original_basis[i].dag()*H_nl*original_vector).data.toarray()[0, 0])*original_basis[i]/(evalue-energy0[i])
                    else:
                        print('The following vector has eigenvalue too close to the original:',multi_indices[i])
                return (new_vector + original_vector)/(new_vector + original_vector).norm()
                '''this function calculates the normalized vector with the first order correction term
                   from the non-linear hamiltonian '''

            d = {i:d.get(i,0) for i in range(N)}
            print('1st Order PT analysis of',d)
            vector0 = fock_state_on(d)
            [multi_indices, basis0, evalues0] = get_basis0(remove=d)
            evalue0 = get_expect_number(vector0, H_lin, vector0)
            vector1 = PT_on_vector(vector0, basis0, evalues0, evalue0)
            evalue1 = get_expect_number(vector0, H_lin+H_nl, vector0)
            evalue2 = get_expect_number(vector1,H_lin+H_nl,vector1)

            if not use_1st_order['return_2O_PT']:
                index = np.argmax([(vector1.dag() * evec).norm() for evec in evecs])
                print("Best ND eigenvector match:",index)
                return evals[index], evecs[index]
            else:
                if d ==  {i:0 for i in range(N)}:
                    return evalue2.real, vector1
                else:
                    return evalue2.real-f0, vector1
    else:
        def closest_state_to(d):
            s = fock_state_on(d)
            def distance(s2):
                return (s.dag() * s2[1]).norm()
            return max(zip(evals, evecs), key=distance)

    
    f0 = closest_state_to({})[0]
    if not use_1st_order['return_2O_PT']:
        evals -= f0
    
    f1s = [closest_state_to({i: 1})[0] for i in range(N)]
    chis = [[0]*N for _ in range(N)]
    chips = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            d = {k: 0 for k in range(N)}       # put 0 photons in each mode (k)
            d[i] += 1
            d[j] += 1
            # load ith mode and jth mode with 1 photon
            # fs = fock_state_on(d)
            ev, evec = closest_state_to(d)
            chi = (ev - (f1s[i] + f1s[j]))
            chis[i][j] = chi
            chis[j][i] = chi

            if chi_prime:
                d[j] += 1
                # fs = fock_state_on(d)
                ev, evec = closest_state_to(d)
                chip = (ev - (f1s[i] + 2*f1s[j]) - 2 * chis[i][j])
                chips[i][j] = chip
                chips[j][i] = chip

    if chi_prime:
        return np.array(f1s), np.array(chis), np.array(chips), np.array(fzpfs), np.array(f0s)
    else:
        return np.array(f1s), np.array(chis), np.array(fzpfs), np.array(f0s)


def black_box_hamiltonian_nq(freqs, zmat, ljs, cos_trunc=6, fock_trunc=8, show_fit=False):
    """
    N-Qubit version of bbq, based on the full Z-matrix
    Currently reproduces 1-qubit data, untested on n-qubit data
    Assume: Solve the model without loss in HFSS.
    """
    nf = len(freqs)
    nj = len(ljs)
    assert zmat.shape == (nf, nj, nj)

    imY = (1/zmat[:, 0, 0]).imag
    # zeros where the sign changes from negative to positive

    (zeros,) = np.where((imY[:-1] <= 0) & (imY[1:] > 0))
    nz = len(zeros)

    imYs = np.array([1 / zmat[:, i, i] for i in range(nj)]).imag
    f0s = np.zeros(nz)
    slopes = np.zeros((nj, nz))
    import matplotlib.pyplot as plt
    # Fit a second order polynomial in the region around the zero
    # Extract the exact location of the zero and the assocated slope
    # If you need better than second order fit, you're not sampling finely enough
    for i, z in enumerate(zeros):
        f0_guess = (freqs[z+1] + freqs[z]) / 2
        zero_polys = np.polyfit(
            freqs[z-1:z+3] - f0_guess, imYs[:, z-1:z+3].transpose(), 2)
        zero_polys = zero_polys.transpose()
        f0s[i] = f0 = min(np.roots(zero_polys[0]),
                          key=lambda r: abs(r)) + f0_guess
        for j, p in enumerate(zero_polys):
            slopes[j, i] = np.polyval(np.polyder(p), f0 - f0_guess)
        if show_fit:
            plt.plot(freqs[z-1:z+3] - f0_guess, imYs[:, z-1:z +
                                                     3].transpose(), lw=1, ls='--', marker='o', label=str(f0))
            p = np.poly1d(zero_polys[0, :])
            p2 = np.poly1d(zero_polys[1, :])
            plt.plot(freqs[z-1:z+3] - f0_guess, p(freqs[z-1:z+3] - f0_guess))
            plt.plot(freqs[z-1:z+3] - f0_guess, p2(freqs[z-1:z+3] - f0_guess))
            plt.legend(loc=0)

    zeffs = 2 / (slopes * f0s[np.newaxis, :])
    # Take signs with respect to first port
    zsigns = np.sign(zmat[zeros, 0, :])
    fzpfs = zsigns.transpose() * np.sqrt(hbar * abs(zeffs) / 2)

    H = black_box_hamiltonian(f0s, ljs, fzpfs, cos_trunc, fock_trunc)
    return make_dispersive(H, fock_trunc, fzpfs, f0s)

black_box_hamiltonian_nq = black_box_hamiltonian_nq