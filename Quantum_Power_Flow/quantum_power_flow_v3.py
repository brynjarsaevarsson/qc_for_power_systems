# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:03:22 2021

@author: brysa
"""
import sys
sys.path.append('..')

import numpy as np
import scipy.linalg as la
from power_flow import PowerFlowNewton, PowerFlowFD, PowerFlowDC
import Classical_Functions.power_systems as ps

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit import Aer
from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.compiler import transpile
from typing import Optional, Union, List

from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal

from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

from qiskit_ibm_provider import IBMProvider

my_provider = IBMProvider()

# Load power system
# bus, branch, gen = ps.IEEE_5bus_qpf()
bus, branch, gen = ps.IEEE_3bus_qpf()
# bus, branch, gen = ps.IEEE_9bus_qpf()


# Construct matrices
Ybus, V0, Sbus, pq_index, pv_index, ref, Y_from, Y_to = ps.make_Ybus(bus, branch)
Bp, Bpp = ps.make_Bmat(bus, branch)

print('Eigenvalues of Bp: ', np.linalg.eigvals(Bp))

# Perform classical power flow for comparison
max_iter = 100
err_tol = 1e-4

V1, success1, n1 = PowerFlowNewton(Ybus, Sbus, V0, ref, pv_index, pq_index, max_iter, err_tol)
V2, success2, n2 = PowerFlowFD(Ybus, Sbus, V0, ref, pv_index, pq_index, max_iter, err_tol)

B, Bf, Pbusinj, Pfinj = ps.make_Bdc(bus, branch)
V3, success3, PF = PowerFlowDC(B, Bf, Pbusinj, Pfinj, bus['Gsh'], bus['Va'], Sbus, ref, pv_index, pq_index)

if not success1:
    print('NO CONVERGENCE NR')

#%% 

# simplified version of the HHL implementation from qiskit # qiskit.algorithms.linear_solvers.hhl
# spit into two functions so only the input part of the HHL circuit needs to be updated

def get_delta(n_l: int, lambda_min: float, lambda_max: float) -> float:
    """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.
    Args:
        n_l: The number of qubits to represent the eigenvalues.
        lambda_min: the smallest eigenvalue.
        lambda_max: the largest eigenvalue.
    Returns:
        The value of the scaling factor.
    """
    formatstr = "#0" + str(n_l + 2) + "b"
    lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
    # floating point precision can cause problems
    if np.abs(lambda_min_tilde - 1) < 1e-7:
        lambda_min_tilde = 1
    binstr = format(int(lambda_min_tilde), formatstr)[2::]
    lamb_min_rep = 0
    for i, char in enumerate(binstr):
        lamb_min_rep += int(char) / (2 ** (i + 1))
    return lamb_min_rep

def construct_circuit(
    matrix: Union[List, np.ndarray, QuantumCircuit],
    vector: Union[List, np.ndarray, QuantumCircuit],
    neg_vals: Optional[bool] = True,
    tomography: Optional[bool] = True,
    ol: Optional[int] = 3,
) -> QuantumCircuit:
    
    nb = int(np.ceil(np.log2(len(vector))))
    vector_circuit = QuantumCircuit(nb,name='input')
    
    if all(vector==0):
        vector+=1e-6
    
    vector_circuit.isometry(vector / np.linalg.norm(vector), list(range(nb)), None)
    
    matrix_circuit = NumPyMatrix(matrix, evolution_time=2*np.pi)
    
    # Set the tolerance for the matrix approximation
    if hasattr(matrix_circuit, "tolerance"):
        matrix_circuit.tolerance = 1e-2 / 6

    # check if the matrix can calculate the condition number and store the upper bound
    if (
        hasattr(matrix_circuit, "condition_bounds")
        and matrix_circuit.condition_bounds() is not None
    ):
        kappa = matrix_circuit.condition_bounds()[1]
    else:
        kappa = 1
    # Update the number of qubits required to represent the eigenvalues
    # The +neg_vals is to register negative eigenvalues because
    # e^{-2 \pi i \lambda} = e^{2 \pi i (1 - \lambda)}
    nl = max(nb + 1, int(np.log2(kappa)) + 1) + neg_vals

    # check if the matrix can calculate bounds for the eigenvalues
    if hasattr(matrix_circuit, "eigs_bounds") and matrix_circuit.eigs_bounds() is not None:
        lambda_min, lambda_max = matrix_circuit.eigs_bounds()
        # Constant so that the minimum eigenvalue is represented exactly, since it contributes
        # the most to the solution of the system. -1 to take into account the sign qubit
        delta = get_delta(nl - neg_vals, lambda_min, lambda_max)
        # Update evolution time
        matrix_circuit.evolution_time = 2 * np.pi * delta / lambda_min / (2**neg_vals)
        # Update the scaling of the solution
        scaling = lambda_min
    else:
        delta = 1 / (2**nl)
        print("The solution will be calculated up to a scaling factor.")

    reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)

    # Initialise the quantum registers
    qb = QuantumRegister(nb,'nb')  # right hand side and solution
    ql = QuantumRegister(nl,'nl')  # eigenvalue evaluation qubits

    qf = QuantumRegister(1,'na')  # flag qubits


    qc = QuantumCircuit(qb, ql, qf)
    
    ic = QuantumCircuit(qb, ql, qf)
    # State preparation
    ic.append(vector_circuit, qb[:])
    qc.barrier()
    # QPE
    phase_estimation = PhaseEstimation(nl, matrix_circuit)

    qc.append(phase_estimation, ql[:] + qb[:])

    # Conditioned rotation
    qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])

    # QPE inverse
    qc.append(phase_estimation.inverse(), ql[:] + qb[:])
    
    if tomography:
        qubits = np.arange(nb+1).tolist()
        qubits[-1] = nb+nl # TODO it is probably not necessary to perform tomography on the ancilla qubit but the current code can't run without it
        qc = state_tomography_circuits(qc, qubits)

        qc = transpile(qc,basis_gates=['id', 'rz', 'sx', 'x', 'cx'], optimization_level=ol)
        ic.add_register(qc[0].cregs[0])
    else:
        qc = transpile(qc,basis_gates=['id', 'rz', 'sx', 'x', 'cx'], optimization_level=ol)

    return qc, ic

def update_circuit(
        qc: Union[List, np.ndarray, QuantumCircuit],
        ic: Union[List, np.ndarray, QuantumCircuit],
        vec: Union[List, np.ndarray, QuantumCircuit],
        ol: Optional[int] = 1,
        ) -> QuantumCircuit:
    """Update the input circuit and combine it with the constant part of the HHL circuit.
    Args:
        matrix: The matrix specifying the system, i.e. A in Ax=b.
        vector: The vector specifying the right hand side of the equation in Ax=b.
        neg_vals: States whether the matrix has negative eigenvalues. If False the
        computation becomes cheaper.
    Returns:
        The HHL circuit.
    Raises:
        ValueError: If the input is not in the correct format.
        ValueError: If the type of the input matrix is not supported.
    """
    
    # Remove the old vector_circuit
    if len(ic.data)>0:
        ic.data.pop(0)
    
    # Create a new vector_circuit and add it to the input 
    nb = int(np.log2(len(vec)))
    vector_circuit = QuantumCircuit(nb,name='input')
    vector_circuit.isometry(vec / np.linalg.norm(vec), list(range(nb)), None)
    
    ic.append(vector_circuit,range(nb))

    if type(qc) == list: # including state tomography 
        circuit = []
        for circ in qc:
            # c = ic+circ # XXX this is deprecated.
            ind = [ic.qubits.index(q) for q in circ.qubits]
            c = ic.compose(circ,qubits = [ic.qubits[i] for i in ind])
            
            c.name = circ.name
            circuit.append(c)
    else:    
        circuit = ic.compose(qc)
        cr = ClassicalRegister(nb+1, 'c')
        circuit.add_register(cr)
        circuit.barrier()
        qubits = np.arange(nb+1).tolist()
        qubits[-1] = qc.num_qubits-1
        circuit.measure(qubits,np.arange(nb+1).tolist())
    
    return circuit



#%% Function for solving HHL

def solve_hhl(mat, vec, hhl_cir, ic, backend, tomography=True, Shots = 2**12):
    
    bNorm = np.linalg.norm(vec)

    true_solution = np.linalg.solve(mat, vec if neg_vals else -vec) # XXX when tomography is not used we get the sign of the result classically

    num_qubits = int(np.log2(vec.shape[0])) # number of variables (assuming n=2^k)

    hhl_circuit = update_circuit(hhl_cir, ic, vec)
    
    n_var = len(vec)
    approx_solution = np.zeros(n_var)
    
    if tomography:
        
        depth = hhl_circuit[0].depth()
        ops = hhl_circuit[0].count_ops()
        
        
        job = execute(hhl_circuit, backend = backend, shots = Shots)
        tomo = StateTomographyFitter(job.result(), hhl_circuit)
        
        rho = tomo.fit()
        
        amp = np.zeros(len(vec),dtype=complex)
        signs = np.zeros(len(vec),dtype=complex)
        for i, a in enumerate(rho):
    
            # get binary representation
            b = ("{0:0%sb}" % (num_qubits+1)).format(i)
             
            i_normal = int(b[-num_qubits:], 2)
            if int(b[0], 2) == 1:
    
                amp[i_normal] += bNorm*np.sqrt(a[i]).real
    
                signs[i_normal] += rho[2**(num_qubits),i]
            
            
        approx_solution = amp.real*np.sign(signs.real)*np.sign(vec[0])
        
        if np.all(np.sign(approx_solution)==np.sign(true_solution)):
            print('True')
        else:
            print('False')
            print('vec: ',vec)
            print('amp: ',approx_solution)
            print('true: ', true_solution)
            
    else:
        
        circuitToRun = transpile(hhl_circuit,basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        
        depth = circuitToRun.depth()
        ops = circuitToRun.count_ops()
        result_indx = np.arange(2 ** (num_qubits), 2 ** (num_qubits+1))

        job = execute(circuitToRun, backend = backend, shots = Shots)
    
        job_result = job.result()
        M = job_result.get_counts()
    
       
        for i in range(n_var):
            try:
                approx_solution[i] = bNorm*np.sqrt(M[str(np.binary_repr(result_indx[i]))]/Shots)*np.sign(true_solution[i])
            except:
                pass
            finally:
                pass
    
    return approx_solution, job_result, depth, ops

#%% Quantum Power Flow

# real_devices = ['ibmq_lima','ibmq_belem','ibmq_bogota','ibmq_manila','ibmq_quito']

backend = Aer.get_backend('qasm_simulator')

dQ = pq_index
dV = pq_index

dP = np.concatenate((pv_index, pq_index))
dTheta = np.concatenate((pv_index, pq_index))

neg_vals = False # Using negative eigenvalues requires more qubits and gates
tomography = False # Tomography is used to extract the full quantum state but requires more runs. If False then we need to find the sign of the result classically

success = False

n = 0

Vm = abs(V0)
Va = np.angle(V0)

err_tol = 1e-5
max_iter = 100
Vm_iter = []
Th_iter = []
dp_iter = []
dq_iter = []

jobs = []

depth_p = []

times= []
gates_p = []

mis = (Sbus - V0 * np.conj(Ybus @ V0))/ Vm
dfp = np.real(mis[dP])
dfq = np.imag(mis[dQ])

# Create the initial HHL circuit # Bp == Bpp
if not neg_vals:
    hhl_cir, ic = construct_circuit(-Bp, -dfp, neg_vals=neg_vals, tomography=tomography)
else:
    hhl_cir, ic = construct_circuit(Bp, dfp, neg_vals=neg_vals, tomography=tomography)

while not success and n < max_iter:

    n += 1
    vec = dfp if neg_vals else -dfp
    approx_dVa, job_result, depth, ops = solve_hhl(Bp, vec, hhl_cir, ic, backend, tomography=tomography)

    classic_dVa = la.solve(Bp, dfp)
    dVa = -approx_dVa

    Va[dP]  = Va[dP]  + dVa
    V = Vm * np.exp(1j * Va)

    mis = (Sbus - V * np.conj(Ybus @ V)) / Vm
    dfp = np.real(mis[dP])
    dfq = np.imag(mis[dQ])

    normP = la.norm(dfp, np.inf)
    dp_iter.append(normP)
    if len(pq_index) > 0:
        normQ = la.norm(dfq, np.inf)
        
    if normP < err_tol and normQ < err_tol:
        success = True
        break

    if len(pq_index) > 0:
        vec = dfq if neg_vals else -dfq
        approx_dVm, job_result, depth, ops = solve_hhl(Bpp, vec, hhl_cir, ic, backend, tomography=tomography)
        dVm = -approx_dVm,
        classic_dVm = la.solve(Bpp, dfq)
    else:
        dVm = 0

    Vm[dQ] = Vm[dQ] + dVm
    V = Vm * np.exp(1j * Va)

    mis = -(V * np.conj(Ybus @ V) - Sbus) / Vm
    dfp = np.real(mis[dP])
    dfq = np.imag(mis[dQ])

    normP = la.norm(dfp, np.inf)
    if len(pq_index) > 0:
        normQ = la.norm(dfq, np.inf)
        dq_iter.append(normQ)
        
    print(n,normP)
    if normP < err_tol and normQ < err_tol:
        success = True
        break

print('QC',V)
print('CC',V2)
# print('Equal results:',np.allclose(V, V2.T, rtol=err_tol, atol=1e-08))

