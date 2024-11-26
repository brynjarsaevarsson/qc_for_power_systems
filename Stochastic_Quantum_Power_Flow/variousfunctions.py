# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:35:48 2023

@author: brysa
"""

import numpy as np
import scipy as sp
import numpy.ctypeslib as npct
import ctypes
import sys, os
from scipy.sparse import csc_matrix, linalg as sla
import copy
from qiskit.quantum_info import partial_trace, Statevector
dll_path = os.path.join(os.path.dirname(__file__), 'libamd.dll')

def isunitary_old(*mat):
    '''
    Check if a matrix is unitary

    Parameters
    ----------
    mat : np.array
        One or more arrays

    Returns
    -------
    bool or list of bools
        True if the matrix is unitary.

    '''
    if len(mat) > 1:
        return [np.allclose(np.eye(len(m)), m.dot(m.T.conj())) if issquare(m) else False for m in mat]
    elif issquare(mat[0]):
        return np.allclose(np.eye(len(mat[0])), mat[0].dot(mat[0].T.conj()))
    else:
        return False
    
    
def isunitary(*mat):
    '''
    Check if a matrix is unitary

    Parameters
    ----------
    mat : np.array
        One or more arrays

    Returns
    -------
    bool or list of bools
        True if the matrix is unitary.

    '''
    return isidentity(*[m@m.T.conj() if issquare(m) else np.zeros((1,1)) for m in mat])
    
    
def isidentity(*mat):
    '''
    Check if a matrix is an identity matrix

    Parameters
    ----------
    mat : np.array
        One or more arrays

    Returns
    -------
    bool or list of bools
        True if the matrix is an identity matrix.

    '''
    if len(mat) > 1:
        return [np.allclose(np.eye(len(m)), m) if issquare(m) else False for m in mat]
    elif issquare(mat[0]):
        return np.allclose(np.eye(len(mat[0])), mat[0])
    else:
        return False
    

def ishermitian(*mat):
    '''
    Check if a matrix is hermitian

    Parameters
    ----------
    mat : np.array
        One or more arrays

    Returns
    -------
    bool or list of bools
        True if the matrix is hermitian.

    '''
    if len(mat) > 1:
        return [np.allclose(m, m.T.conj()) if issquare(m) else False for m in mat]
    elif issquare(mat[0]):
        return np.allclose(mat[0], mat[0].T.conj())
    else:
        return False

def issymmetric(mat):
    '''
    Check if a matrix is symmetric

    Parameters
    ----------
    mat : np.array

    Returns
    -------
    bool
        True if the matrix is symmetric.

    '''
    if not issquare(mat): return False
    
    return np.allclose(mat, mat.T)

def issquare(mat):
    '''
    Check if a matrix is square

    Parameters
    ----------
    mat : np.array

    Returns
    -------
    bool
        True if the matrix is square.

    '''
    return mat.shape[0] == mat.shape[1] and len(mat.shape) == 2

def hermitian_matrix(mat):
    '''
    Modify a matrix to be hermitian 
    Hmat = [[0, M]
            [M.T*,0]]

    Parameters
    ----------
    mat : np.array

    Returns
    -------
    np.array
        Hermitian matrix

    '''
    
    
    zm1 = np.zeros((mat.shape[0],mat.shape[0]))
    zm2 = np.zeros((mat.shape[1],mat.shape[1]))

    Hmat = np.vstack((np.hstack((zm1,mat)),np.hstack((mat.T.conj(),zm2))))
    
    return Hmat

def decompose_to_unitary(A,forcehermitian=False):
    '''
    Decompose a hermitian matrix with a norm=1 to four unitary matrices
    UB, VB, UC, VC so that A = (UB+VB)/2 + 1j*(UC+VC)/2

    Parameters
    ----------
    A : np.array
        Hermitian matrix with norm=1

    Returns
    -------
    UB: np.array
        Unitary matrix.
    VB: np.array
        Unitary matrix.
    UC: np.array
        Unitary matrix.
    VC: np.array
        Unitary matrix.

    '''
    if not ishermitian(A) and forcehermitian:
        A = hermitian_matrix(A)/np.sqrt(2)
        
    if not ishermitian(A) or not round(np.linalg.norm(A),6)==1:
        print('Matrix must be Hermitian and with norm=1')
        return 0, 0, 0, 0
    # Decompose a complex matrix into unitary matrices
    B = 1/2*(A+A.conj()) # A.real
    C = 1/(2j)*(A-A.conj()) # A.imag

    UB = B +1j*sp.linalg.sqrtm(np.eye(B.shape[0])-B@B)
    VB = B -1j*sp.linalg.sqrtm(np.eye(B.shape[0])-B@B)
    
    UB[np.where(abs(UB)<1e-9)] = 0
    VB[np.where(abs(VB)<1e-9)] = 0

    UC = C + 1j*sp.linalg.sqrtm(np.eye(C.shape[0])-C@C)
    VC = C - 1j*sp.linalg.sqrtm(np.eye(C.shape[0])-C@C)
    
    UC[np.where(abs(UC)<1e-9)] = 0
    VC[np.where(abs(VC)<1e-9)] = 0
    
    return UB, VB, UC, VC

def pad_to_n_qubits(A, n=None, d=1):
    '''
    Pad a matrix to size 2**n with ones on the diagonal

    Parameters
    ----------
    A : np.array
        input matrix.
    n : int, optional
        the desired size of the matrix as 2**n. If None then it pads to the 
        nearest value of 2**n for the given matrix

    Returns
    -------
    Apad : np.array
        a padded array of size 2**n.

    '''
    size_mat = len(A)
    
    if n is None:
        n = int(np.ceil(np.log2(size_mat)))

    Apad = np.pad(A,(0,int(2**n)-size_mat))
    Apad[np.arange(size_mat,int(2**n)),np.arange(size_mat,int(2**n))] = d
    
    return Apad

def pad_to_n_qubits2(A, n=None):
    '''
    Pad a matrix to size 2**n with ones on the diagonal

    Parameters
    ----------
    A : np.array
        input matrix.
    n : int, optional
        the desired size of the matrix as 2**n. If None then it pads to the 
        nearest value of 2**n for the given matrix

    Returns
    -------
    Apad : np.array
        a padded array of size 2**n.

    '''
    dim1 = A.shape[0]
    dim2 = A.shape[1]
    
    if n is None:
        n = int(np.ceil(np.log2(max(dim1,dim2))))

    Apad = np.pad(A,((0,int(2**n)-dim1),(0,int(2**n)-dim2)))
    Apad[np.arange(dim1,int(2**n)),np.arange(dim1,int(2**n))-4] = 1
    
    return Apad

def compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=None):
    '''
    Add a control qubit to a unitary matrix. This function is taken from
    Qiskit source code.

    Parameters
    ----------
    base_mat : np.array
        matrix to be controlled.
    num_ctrl_qubits : int
        number of control qubits.
    ctrl_state : str, optional
        '0' or '1' The value of the ctrl qubit to apply the control action. The default is None.

    Returns
    -------
    full_mat : np.array
        matrix with control action applied.

    '''
    num_target = int(np.log2(base_mat.shape[0]))
    ctrl_dim = 2**num_ctrl_qubits
    ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim - 1])
    if ctrl_state is None:
        ctrl_state = ctrl_dim - 1
    elif isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state, 2)
    
    ctrl_proj = np.diag(np.roll(ctrl_grnd, ctrl_state))
    full_mat = np.kron(np.eye(2**num_target), np.eye(ctrl_dim) - ctrl_proj) + np.kron(
        base_mat, ctrl_proj
    )
    return full_mat


def expand_system(vec, mat):
    '''
    Flatten the bus injection matrix to a vector and expand the 
    bus-line matrix accordingly to get a vector of line loadings

    Parameters
    ----------
    vec : np.array
        Nbus by Nbin matrix with power injection distributions on each bus.
    mat : np.array
        Nbranch by Nbus DC power flow matrix for mapping bus injections to line flows.

    Returns
    -------
    flat_vec : np.array
        Nbus*Nbin by 1 vector with power injection distributions on each bus.
    exp_mat : np.array
        Expanded Nbranch*Nbin by Nbus*Nbin DC power flow matrix for mapping a 
        vector bus injections to a line flow vector

    '''
    
    
    nbus = vec.shape[0]
    nbr = mat.shape[0]
    nbin = vec.shape[1]
    
    lbus = nbus*nbin
    lbr = nbr*nbin
    
    exp_mat = np.zeros((lbr,lbus))
    
    flat_vec = vec.flatten()
    
    for i, row in enumerate(mat):
        k = i*nbin
        for m, val in enumerate(row):
            p = m*nbin
            exp_mat[k:k+nbin,p:p+nbin] = np.eye(nbin)*val
            
    return flat_vec, exp_mat

def expand_matrix_old(mat, nqb):

    nbus = mat.shape[1]
    nbr = mat.shape[0]
    nbin = 2**nqb

    Na = 2**(nbus*nqb)

    exp_mat = np.zeros((Na,Na))
    
    Mcp = np.zeros((Na,Na),dtype=int)
    Mcdbl = np.zeros((Na,Na),dtype=int)
    Mcm = np.zeros((Na,Na),dtype=int)

    for i in range(nbin): # XXX        
        inds1 = np.arange(i,nbin+i)
        inds2 = np.arange(i*nbin,(i+1)*nbin)
        Mcp[inds1,inds2] = 1
        
        inds3 = np.abs(np.arange(nbin)-i)
        inds4 = np.arange(i*nbin,(i+1)*nbin)
        Mcm[inds3,inds4] = 1
        
        inds5 = np.arange(i*2,nbin+i*2)
        inds6 = np.arange(i*nbin,(i+1)*nbin)
        Mcdbl[inds5,inds6] = 1
        
    # TODO this only works for the test matrix M = np.array([[2., 1.],[2., 1.], [1., -1.]])
    for i, row in enumerate(mat):
        if np.all(np.sign(row)==np.sign(row[0])):
            if row[0] == row[1]:
                exp_mat += Mcp
            else:
                exp_mat += Mcdbl # TODO this is only when row[0] == 2*row[1]
        else:
            exp_mat += Mcm
                
    return exp_mat/nbr

def expand_matrix_scale2(PTDF, nqb):
    if PTDF.ndim > 1:
        PTDF = PTDF[0]

    nbus = len(PTDF)
        
    nbin = 2**nqb
    Na = 2**(nbus*nqb)

    # nbus = len(PTDF) 
    sm = np.zeros(nbin**nbus)
    
    r = np.arange(nbin)

    # I = np.ones(nbin)
    # for bus in range(nbus):
    #     wvec = PTDF[bus]*r
    #     for k in range(nbus-1):
    #         if k < bus:
    #             wvec = np.kron(wvec,I)
    #         else:
    #             wvec = np.kron(I,wvec)
    #     sm += wvec
        
    sm = np.zeros(1)
    for bus in range(nbus):
        wvec = PTDF[bus]*r
        sm = kronsum(wvec,sm)
    sm = abs(sm)
    unique = np.unique(sm)
    Nu = len(unique)
    
    M = np.zeros((Nu,Na))
    sv = np.zeros(Nu)
    for i, v in enumerate(unique):
        vals = np.flatnonzero(sm==v)
        M[i,vals] = 1
        sv[i] = vals.size
    
    sv = np.expand_dims(sv, 1)
    rootsum = np.sqrt(sv)
    
    Msc = M/rootsum
    
    # m2 = (np.eye(Na)-Msc.T@Msc).T
    
    # ovs = np.flatnonzero(np.sum(m2,axis=1))
    # nrms = np.expand_dims(np.linalg.norm(m2[ovs],axis=1),1)
    
    # vecs = m2[ovs]/nrms
    # # v2 = m2[ovs[1]]/np.linalg.norm(m2[ovs[1]])
    # Msc2 = np.vstack((Msc,vecs))
    # # Msc2 = np.vstack((Msc2,v2))
    
    return Msc, unique, rootsum

def expand_matrix_scale3(PTDF, nqb):
    # if PTDF.ndim > 1:
    #     PTDF = PTDF[0]
    
    nbr = PTDF.shape[0]
    nbus = PTDF.shape[1]
        
    nbin = 2**nqb
    Na = 2**(nbus*nqb)
    MM = []
    U = []
    MAT = np.zeros((136,16))
    for br in range(nbr):
        # nbus = len(PTDF) 
        sm = np.zeros(nbin**nbus)
        
        r = np.arange(nbin)
    
        # I = np.ones(nbin)
        # for bus in range(nbus):
        #     wvec = PTDF[bus]*r
        #     for k in range(nbus-1):
        #         if k < bus:
        #             wvec = np.kron(wvec,I)
        #         else:
        #             wvec = np.kron(I,wvec)
        #     sm += wvec
            
        sm = np.zeros(1)
        for bus in range(nbus):
            wvec = PTDF[br,bus]*r
            sm = kronsum(wvec,sm)
        sm = abs(sm)
        unique = np.unique(sm)
        Nu = len(unique)
        
        M = np.zeros((Nu,Na))
        sv = np.zeros(Nu)
        for i, v in enumerate(unique):
            vals = np.flatnonzero(sm==v)
            M[i,vals] = 1
            MAT[int(v),vals] += 1
            sv[i] = vals.size
        
        sv = np.expand_dims(sv, 1)
        rootsum = np.sqrt(sv)
        
        Msc = M/rootsum
        MM.append(Msc)
        u,s,vh = np.linalg.svd(Msc)
        print(unique)
        U.extend(unique)
        
    Unique = np.unique(U).astype(int)
    
    MAT = MAT[Unique]#[:14,:]
    
    x,y = sp.Matrix(MAT).T.rref()
    
    MAT = MAT[np.array(y),:]
    
    u,s,vh = np.linalg.svd(MAT/np.linalg.norm(MAT,axis=1).reshape(-1,1))
    
    # m2 = (np.eye(Na)-Msc.T@Msc).T
    
    # ovs = np.flatnonzero(np.sum(m2,axis=1))
    # nrms = np.expand_dims(np.linalg.norm(m2[ovs],axis=1),1)
    
    # vecs = m2[ovs]/nrms
    # # v2 = m2[ovs[1]]/np.linalg.norm(m2[ovs[1]])
    # Msc2 = np.vstack((Msc,vecs))
    # # Msc2 = np.vstack((Msc2,v2))
    
    return Msc, unique, rootsum

def expand_matrix_scale(PTDF, nqb):
    if PTDF.ndim > 1:
        PTDF = PTDF[0]

    nbus = len(PTDF)
        
    nbin = 2**nqb
    Na = 2**(nbus*nqb)

    nbus = len(PTDF) 
    sm = np.zeros(nbin**nbus)
    
    r = np.arange(nbin)

    I = np.ones(nbin)
    for bus in range(nbus):
        wvec = PTDF[bus]*r
        for k in range(nbus-1):
            if k < bus:
                wvec = np.kron(wvec,I)
            else:
                wvec = np.kron(I,wvec)
        sm += wvec
        
    # sm = np.zeros(1)
    # for bus in range(nbus):
    #     wvec = PTDF[bus]*r
    #     sm = kronsum(wvec, sm)
    
    sm = abs(sm)
    unique = np.unique(sm)
    Nu = len(unique)
    
    M = np.zeros((Nu,Na))
    sv = np.zeros(Nu)
    for i, v in enumerate(unique):
        vals = np.flatnonzero(sm==v)
        M[i,vals] = 1
        sv[i] = vals.size
    
    sv = np.expand_dims(sv, 1)
    rootsum = np.sqrt(sv)
    
    Msc = M/rootsum
    
    # m2 = (np.eye(Na)-Msc.T@Msc).T
    
    # ovs = np.flatnonzero(np.sum(m2,axis=1))
    # nrms = np.expand_dims(np.linalg.norm(m2[ovs],axis=1),1)
    
    # vecs = m2[ovs]/nrms
    # # v2 = m2[ovs[1]]/np.linalg.norm(m2[ovs[1]])
    # Msc2 = np.vstack((Msc,vecs))
    # # Msc2 = np.vstack((Msc2,v2))
    
    return Msc, unique, rootsum

def expand_matrix(mat, nqb, rmzeros=False):
    def build_mat(weights, nbin):
        nbus = len(weights) 
        # out = np.zeros(nbin**nbus - (nbus-1))
        
        r = np.arange(nbin)

        # I = np.ones(nbin)
        # for bus in range(nbus):
        #     wvec = weights[bus]*r
        #     for k in range(nbus-1):
        #         if k < bus:
        #             wvec = np.kron(wvec,I)
        #         else:
        #             wvec = np.kron(I,wvec)
        #     sm += wvec
        # for i in range(len(out)):
        #     out[i] = weights[0]
        # out = np.unique(np.concatenate([abs(r*w) for w in weights]))
  
        sm = np.zeros(1,dtype=int)
        for bus in range(nbus):
            wvec = int(weights[bus])*r
            sm = kronsum(wvec, sm)
            
        sm = abs(sm)
        out = np.unique(sm)
        mx = int(np.max(sm))
        tmp_mat = np.zeros((Nk,Na))
        # for i in range(mx+1):
        #     # if len(np.flatnonzero(sm==i))>0:
        #     #     print(np.flatnonzero(sm==i),i)
        #     tmp_mat[i,np.flatnonzero(sm==i)] = 1
            
        for i in out:
            tmp_mat[i,np.flatnonzero(sm==i)] = 1
        return tmp_mat

    if mat.ndim == 1:
        mat = np.expand_dims(mat,0)
    # else:
    nbus = mat.shape[1]
    nbr = mat.shape[0]
        
    nbin = 2**nqb
    Na = 2**(nbus*nqb)
    
    # mx = int(np.max(abs(np.sum(mat,axis=1))))
    mx = int(np.max([abs(np.sum(mat[0,np.flatnonzero(mat<0)])),np.sum(mat[0,np.flatnonzero(mat>0)])]))
    
    if mx == 0:
        mx = int(np.max(abs(mat))) # TODO I don't think this works for all cases
    # mxv = (nbin-1)*mx+1
    Nk = 2**(int(np.ceil(np.log2((nbin-1)*mx)))) # Set the shape of the array according to the largest value in mat 
    exp_mat = np.zeros((Nk,Na))
    
    for i, row in enumerate(mat): 
        exp_mat += build_mat(row, nbin)
        outp_range = np.arange(Nk)
        
    if rmzeros:
        outp_range = np.flatnonzero(np.sum(exp_mat,axis=1))
        exp_mat = exp_mat[outp_range]
    
    return exp_mat, outp_range #exp_mat/nbr

def expand_matrix_old(mat, nqb, rmzeros=False):
    
    # def loop_rec(a,sm,nbus,w,v=0,cnt=0):
    #     if nbus >= 1:
    #         for x in a:
    #             cnt = loop_rec(a, sm, nbus-1, w, v+x*np.flip(w)[nbus-1],cnt)
    #         return cnt
    #     else:
    #         sm[cnt] = abs(v)
    #         return cnt + 1

    def build_mat_old(weights, nbin):
        # Creates a sum vector using a recursive loop and then builds a matrix 
        # from the indices of the values in the sum vector.
        # XXX there is probably a more efficient way to do this since the 
        # matrix shape looks very predictable
        def loop_rec(a, sm, nbus, w, v=0, cnt=0):
            if nbus >= 1:
                for x in a:
                    cnt = loop_rec(a, sm, nbus-1, w, v+x*np.flip(w)[nbus-1], cnt)
                return cnt
            else:
                sm[cnt] = abs(v)
                return cnt + 1
            
        nbus = len(weights)    
        sm = np.zeros(nbin**nbus)
        
        loop_rec(np.arange(nbin),sm, nbus, weights)
        mx = int(np.max(sm))
        tmp_mat = np.zeros((Nk,Na))
        # sv = []
        # sms = []
        for i in range(mx+1):
            tmp_mat[i,np.flatnonzero(sm==i)] = 1
            # sv.append(list(np.flatnonzero(sm==i)))
            # sms.append(np.sum(sm==i))
            

        # def calc_sm(weights,nbin):
        #     def loop_rec(a, sm, nbus, w, v=0, cnt=0):
        #         if nbus >= 1:
        #             for x in a:
        #                 cnt = loop_rec(a, sm, nbus-1, w, v+x*np.flip(w)[nbus-1], cnt)
        #             return cnt
        #         else:
        #             sm[cnt] = abs(v)
        #             return cnt + 1
                
        #     nbus = len(weights)    
        #     sm = np.zeros(nbin**nbus)
        
        #     loop_rec(np.arange(nbin),sm, nbus, weights)
        #     return sm

        
        return tmp_mat
    
    def build_mat(weights, nbin):
        nbus = len(weights) 
        sm = np.zeros(nbin**nbus)
        
        r = np.arange(nbin)
        # for bus in range(nbus):
        #     # I = np.ones(nbin**(nbus-1))
        #     # I[0] = 1
        #     # sm += np.kron(weights[bus]*r,I)
            
        #     k = nbin**(bus)
        #     for s in range(k):
        #         # sm2[s*nbin**(nbus-bus-1):(s+1)*nbin**(nbus-bus-1)] += weights[bus]*np.mod(s,nbin)
        #         for v in r:
        #             sm[(v+s*nbin)*nbin**(nbus-bus-1):(v+s*nbin+1)*nbin**(nbus-bus-1)] += weights[bus]*v
        
        I = np.ones(nbin)
        for bus in range(nbus):
            wvec = weights[bus]*r
            for k in range(nbus-1):
                if k < bus:
                    wvec = np.kron(wvec,I)
                else:
                    wvec = np.kron(I,wvec)
            sm += wvec
        sm = abs(sm)
        mx = int(np.max(sm))
        tmp_mat = np.zeros((Nk,Na))
        for i in range(mx+1):
            tmp_mat[i,np.flatnonzero(sm==i)] = 1
        return tmp_mat

    if mat.ndim == 1:
        mat = np.expand_dims(mat,0)
    # else:
    nbus = mat.shape[1]
    nbr = mat.shape[0]
        
    nbin = 2**nqb
    Na = 2**(nbus*nqb)
    
    # mx = int(np.max(abs(np.sum(mat,axis=1))))
    mx = int(np.max([abs(np.sum(mat[0,np.flatnonzero(mat<0)])),np.sum(mat[0,np.flatnonzero(mat>0)])]))
    
    if mx == 0:
        mx = int(np.max(abs(mat))) # TODO I don't think this works for all cases
    # mxv = (nbin-1)*mx+1
    Nk = 2**(int(np.ceil(np.log2((nbin-1)*mx)))) # Set the shape of the array according to the largest value in mat 
    exp_mat = np.zeros((Nk,Na))
    
    for i, row in enumerate(mat): 
        exp_mat += build_mat(row, nbin)
        outp_range = np.arange(Nk)
        
    if rmzeros:
        outp_range = np.flatnonzero(np.sum(exp_mat,axis=1))
        exp_mat = exp_mat[outp_range]
    
    return exp_mat, outp_range #exp_mat/nbr


def amd_order(matrix):
    mat = csc_matrix(matrix)
    # move the imports oto preamble
    lib_amd = ctypes.CDLL(dll_path)

    # used to describe the input to the c functions..
    array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

    # Describe return and arguments datatypes
    lib_amd.amd_order.restype = np.int32
    lib_amd.amd_order.argtypes = [ctypes.c_int, array_1d_int, array_1d_int, array_1d_int, array_1d_double,
                                  array_1d_double]
    lib_amd.amd_defaults.argtypes = [array_1d_double]
    lib_amd.amd_info.argtypes = [array_1d_double]
    lib_amd.amd_control.argtypes = [array_1d_double]

    N_nc = np.int32(mat.shape[0])  # number of NC nodes
    # the AMD ordering
    amd_control = np.zeros(5, dtype=np.double)
    amd_info = np.zeros(20, dtype=np.double)
    P = -np.ones(N_nc, dtype=np.int32)
    status = lib_amd.amd_order(N_nc, mat.indptr, mat.indices, P, amd_control, amd_info)
    
    amdmatrix = np.copy(matrix)

    for k in range(len(P)):
        amdmatrix[k] = matrix[P[k]]
        
    return amdmatrix, P

def vec_to_unitary(v, i=0):
    if v.ndim == 1:
        v = np.expand_dims(v,0)
    dim = v.size
    # Return identity if v is a multiple of e1
    if v[0][0] and not np.any(v[0][1:]):
        return np.identity(dim)
    e1 = np.zeros(dim)
    e1[i] = 1
    w = (v/np.linalg.norm(v) - e1)

    return np.identity(dim) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))




def create_unitary2(v):
    # v=weights[:4]
    if v.ndim == 1:
        v = np.expand_dims(v,0)
    dim = v.size
    # # Return identity if v is a multiple of e1
    # if v[0][0] and not np.any(v[0][1:]):
    #     return np.identity(dim)
    e1 = np.zeros(dim)#-1
    e1[-1] = 1
    w = (v/np.linalg.norm(v) - e1/np.linalg.norm(e1))#.T
    umat = np.identity(dim) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))
    vec = np.array([1,2,3,4])
    vec = np.array([0.0069, 0.0108, 0.009 , 0.0033, 0.0299, 0.0468, 0.039 , 0.0143,
           0.0782, 0.1224, 0.102 , 0.0374, 0.115 , 0.18  , 0.15  , 0.055 ])
    
    umat@vec
    isunitary(umat)
    np.linalg.norm(umat@vec)
    
    e0 = np.ones((1,dim))
    e0[0,-1] = 0
    # e0=v
    e1 = np.zeros(dim)-1
    e1[-1] = 0
    
    w = (e0/np.linalg.norm(e0) - e1/np.linalg.norm(e1))#.T
    umat2 = np.identity(dim) - 2*(np.dot(w.T, w)/np.dot(w, w.T))
    isunitary(umat2)
    0.01034301
    umat2.T@umat@vec
    # array([5.34522484, 0.83868981, 0.67737963, 0.51606944])
    return np.identity(dim) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))
    
def normalize_rows(mat,tol=1e-9):
    scaling = np.linalg.norm(mat,axis=1,keepdims=True)
    scaling[np.where(scaling<tol)] = 1
    return mat/scaling, scaling
    
def kronsum(A, B):
    # Calculate the kronecker sum of two vectors
    In = np.ones(B.size,B.dtype)
    Im = np.ones(A.size,A.dtype)
    return np.kron(A,In) + np.kron(Im,B)


def single_qubit_state(circuit, qubit):
    """Get the statevector for the first qubit, discarding the rest."""
    # get the full statevector of all qubits
    full_statevector = Statevector(circuit)
    
    nqb = circuit.num_qubits
    qbs = list(np.delete(np.arange(nqb), qubit))
    # get the density matrix for the first qubit by taking the partial trace
    partial_density_matrix = partial_trace(full_statevector, qbs)

    # extract the statevector out of the density matrix
    partial_statevector = np.sqrt(np.diagonal(partial_density_matrix))

    return partial_statevector


def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):#break
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)

        Y = np.vstack((Y, X[i,:] - proj.sum(0)))

    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
    
    
def counts_to_probs(counts):
    probs = np.zeros(2**len(list(counts.keys())[0]))
    
    if hasattr(counts,'shots'):
        shots = counts.shots()
    else:
        shots = sum(np.fromiter(counts.values(), dtype=int))
    for c in counts.items():
        probs[int(c[0],2)] = c[1]/shots
    return probs

sv = np.array([ 1.,  4., 10., 20., 31., 40., 44., 40., 31., 20., 10.,  4.,  1.])


PTDF = np.arange(3)
nqb = 2
if PTDF.ndim > 1:
    PTDF = PTDF[0]

nbus = len(PTDF)
    
nbin = 2**nqb
Na = 2**(nbus*nqb)

nbus = len(PTDF) 
sm = np.zeros(nbin**nbus)

r = np.arange(nbin)

I = np.ones(nbin)
for bus in range(nbus):
    wvec = PTDF[bus]*r
    for k in range(nbus-1):
        if k < bus:
            wvec = np.kron(wvec,I)
        else:
            wvec = np.kron(I,wvec)
    sm += wvec
    
    
sm2 = np.zeros(1)
for bus in range(nbus):
    wvec = PTDF[bus]*r
    # sm2 = kronsum(sm2,wvec)
    sm2 = kronsum(wvec,sm2)
    
sm = abs(sm)
unique = np.unique(sm)
Nu = len(unique)

M = np.zeros((Nu,Na))
sv = np.zeros(Nu)
for i, v in enumerate(unique):
    vals = np.flatnonzero(sm==v)
    M[i,vals] = 1
    sv[i] = vals.size
sv
