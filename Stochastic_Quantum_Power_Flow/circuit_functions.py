# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:22:48 2023

@author: brysa
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from variousfunctions import pad_to_n_qubits

def matrix_circuit(matrix, qregs=None):
    # Create a quantum circuit for matrix multiplication
    # This currently only works when assessing flow in one line at a time.
    # We scale the matrix by the square root of the sum of the matrix columns
    # which results in the singular values all becoming 1. The matrix also
    # cannot have any zero rows.
    
    size_mat = matrix.shape
    nb = int(np.ceil(np.log2(size_mat[0])))
    nc = int(np.ceil(np.log2(size_mat[1])))
    
    scaling = np.linalg.norm(matrix,axis=1,keepdims=True)

    # Perform Singular Value Decomposition
    u, s, vh = np.linalg.svd(matrix/scaling)
    
    # Pad the matrices to fit to N qubits
    tmp_u = pad_to_n_qubits(u)
    tmp_vh = pad_to_n_qubits(vh)
    
    if qregs is None:
        # Build the quantum circuit
        mat_circuit = QuantumCircuit(max(nb,nc), name='DC power flow')
    else:
        mat_circuit = QuantumCircuit(*qregs, name='DC power flow')
        
    mat_circuit.unitary(tmp_vh,list(np.arange(nc)),label='Vh')
    mat_circuit.unitary(tmp_u,list(np.arange(nb)),label='u')
    
    return mat_circuit, scaling[:,0]

def input_circuit(vec, qregs=None):
    # Load the power distribution vector onto the qubits
    
    # Pad the vector with zeros to fit to N qubits
    nb = int(np.ceil(np.log2(len(vec))))
    tmp_vec = np.pad(vec,(0,int(2**nb)-len(vec)))
    
    # Normalize
    vec_norm = np.linalg.norm(tmp_vec)
    vector = tmp_vec/vec_norm
    
    # Create the circuit
    if qregs is None:
        vc = QuantumCircuit(nb, name='Bus injection')
    else:
        vc = QuantumCircuit(qregs, name='Bus injection')
        
    vc.isometry(vector, list(range(nb)), None)
    vector_circuit = transpile(vc, basis_gates=["u1","u2","u3","cx"], optimization_level=1)
    
    return vector_circuit, vec_norm