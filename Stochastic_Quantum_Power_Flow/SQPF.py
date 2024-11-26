# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:32:46 2022

@author: brysa
"""

import sys
sys.path.append('..')

import numpy as np
import copy
import matplotlib.pyplot as plt

import Classical_Functions.power_systems as ps
from Classical_Functions.power_flow import PowerFlowNewton, PowerFlowDC

from qiskit.compiler import transpile
from qiskit import QuantumCircuit#, Aer
import qiskit_aer as Aer

from qiskit import QuantumRegister

from circuit_functions import input_circuit, matrix_circuit
from variousfunctions import expand_matrix, vec_to_unitary, counts_to_probs

from qiskit_aer.primitives import Sampler

from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

import string

from qiskit.result import marginal_counts

from qiskit_ibm_runtime import QiskitRuntimeService

import json
from qiskit_ibm_runtime import RuntimeEncoder
from qiskit_ibm_runtime import RuntimeDecoder

# from tikzplotlib import save as tikz_save

np.random.seed(1234)

# Load power system
system = ps.IEEE_3bus_sqpf
# system = ps.IEEE_5bus_sqpf

#
bus, branch, gen = system()

# Construct matrices
Ybus, V0, Sbus, pq_index, pv_index, ref, Y_from, Y_to = ps.make_Ybus(bus, branch)

n_bus = len(bus['Nr'])
n_br = len(branch['From'])

# Perform classical AC power flow
max_iter = 100
err_tol = 1e-6

V1, success1, n1 = PowerFlowNewton(Ybus, Sbus, V0, ref, pv_index, pq_index, max_iter, err_tol)
P_from = (V1[branch['From']]*(Y_from.dot(V1)).conj()).real

if not success1:
    print('NO CONVERGENCE')

# Perform classical DC power flow
B, Bf, Pbusinj, Pfinj = ps.make_Bdc(bus, branch)
V, success, PF0 = PowerFlowDC(B, Bf, Pbusinj, Pfinj, bus['Gsh'], bus['Va'], Sbus, ref, pv_index, pq_index)

ll = (branch['rating']).reshape(n_br,1) # Get the line limits  

PTDF = ps.ptdf(bus,branch,False) # Power Transfer Distribution Factors 
PTDFr = PTDF*100/ll #PTDF scaled for 0-100% line loading

PTDFr[:,pq_index-1] *= -1 # XXX assume that all pq buses have negative injection (TODO the power distributions are currently only defined for positive power injection)

# Define the basis gates used by real hardware
basis_gates=['id', 'rz', 'sx', 'x', 'cx']

#%% 

# Choose the line to be analyzed 
line_index = 0

# The PTDFs must be integers so we round them. This introduces a small error but should become better with more qubits
M = np.round(PTDFr)[line_index] 

nqb = 2 # Number of qubits per distribution
nbin = 2**nqb # Number of bins used to represent the distribution

# Use the same power distribution on each bus. If False then the distributions are centered around the values from Sbus
SAMEDISTRIBUTION = True

# Create labels for the buses
labels = string.ascii_uppercase
labels = ['B{:d}'.format(k+1) for k in np.delete(np.arange(n_bus), ref)]

if SAMEDISTRIBUTION:
    
    # # Create a random test distribution
    x1 = np.random.normal(1.00, 0.7, 10000)
    
    # Convert the distribution into discrete bins
    x2 = np.digitize(x1, np.arange(2**nqb-1))
    
    P1 = np.bincount(x2, minlength=nbin)/10000
    ndec = 2
    P1 = np.round(P1,ndec) # XXX might not sum to 1
    if sum(P1)>1: # Correct rounding errors
        s = sum(P1)-1
        P1[np.argmax(P1)] -= s
    if sum(P1)<1: #
        s = 1-sum(P1)
        P1[np.argmin(P1)] += s
    
    Problist = [P1]
    
    # Create a vector with the kroneker product of the bus distributions for classical comparison
    Pvec = P1
    for i in range(n_bus-2): # XXX I'm using the same distribution at all buses
        Pvec = np.kron(P1,Pvec)
        Problist.append(P1)
        
    # Create the quantum circuit with bus injection distributions
    qa = QuantumRegister(nqb, labels[0]) 
    vc, vn = input_circuit(P1, qa)
    vec_norm = vn
    vqc = QuantumCircuit(qa)
    vqc.append(vc.to_gate(), qa[:])
    
    for i in range(n_bus-2):
        qa = QuantumRegister(nqb, labels[i+1]) 
        vc, vn = input_circuit(P1, qa)
        
        vec_norm *= vn
        vqc.add_register(qa)
        vqc.append(vc.to_gate(), qa[:])
        
else:
    std = 0.7
    Pvec = 1
    vec_norm = 1
    vqc = QuantumCircuit()
    
    Problist = []
    vcl = []
    for b in range(n_bus-1):
        P0 = abs(np.real(np.delete(Sbus,ref)[b]))
        
        # # Create a test distribution
        x1 = np.random.normal(P0, std, 100000)
        x1 = np.round(x1).astype(int)
        x1[np.where(x1>=nbin)] = nbin-1
        x1[np.where(x1<0)] = 0
        P1 = np.bincount(x1, minlength=nbin)/100000
        ndec = 2
        P1 = np.round(P1,ndec) # XXX might not sum to 1
        if sum(P1)>1: #
            s = sum(P1)-1
            P1[np.argmax(P1)] -= s
        if sum(P1)<1: #
            s = 1-sum(P1)
            P1[np.argmin(P1)] += s
            
        Problist.append(P1)
        Pvec = np.kron(P1,Pvec)
    
        # Create the quantum circuit with bus injection distributions
        qa = QuantumRegister(nqb, labels[b]) 
        vc, vn = input_circuit(P1, qa)
        vcl.append(vc)
        vec_norm *= vn
        
        vqc.add_register(qa)
        vqc.append(vc.to_gate(), qa[:])
    
# Expand the PTDF matrix to work with nqb qubit distributions at each bus
exp_matrix, outp_range = expand_matrix(M, nqb, True)

res = (exp_matrix@Pvec)
exact_mu = np.dot(outp_range, res)
print('Exact mean value', exact_mu)

FS = 25

# Plot the power distribution of the first bus 
plt.figure(figsize=(8,8))
plt.bar(np.arange(len(P1)),Problist[0]*100,width=1*0.75)

plt.yticks(fontsize=FS)
plt.xticks(list(np.arange(len(P1))),fontsize=FS)
plt.ylabel('Probability of generation %',fontsize=FS)
plt.xlabel('Generated Power MW',fontsize=FS)
plt.tight_layout()

# Plot the kronecker product of input distributions. This shows the probability of each combination of inputs
plt.figure()
plt.bar(np.arange(len(Pvec)),Pvec*100,width=0.5)

plt.yticks(fontsize=FS)
plt.xticks(visible=False)

plt.ylabel('Probability of scenario %',fontsize=FS)
plt.xlabel('Scenario',fontsize=FS)

#%% Run the bus distribution circuit on real and simulated QC (First step)


# Load saved credentials
service = QiskitRuntimeService() 
                                            
backend_sim = Aer.AerSimulator()


# Run on a simulator
t_circ_sim = transpile(vqc, backend_sim)
t_circ_sim.measure_all()

shots = 2**10
job_sim = backend_sim.run(t_circ_sim,shots=shots)

result_sim = job_sim.result()
counts_sim = result_sim.get_counts()

# # Run on a real hardware
# t_circ_hw = transpile(vqc, backend_hw)
# t_circ_hw.measure_all()

# job_hw = backend_hw.run(t_circ_hw,shots=shots)
# job_monitor(job_hw)
# result_hw = job_hw.result()
# counts_hw = result_hw.get_counts()

# Retrieve the result from the real hardware
# retrieved_job = service.job("cicn96902cefj76f37cg")
# with open("retrieved_job_1.json", "w") as file:
#     json.dump(retrieved_job.result(), file, cls=RuntimeEncoder)
    
with open("retrieved_job_1.json", "r") as file:
    retrieved_job = json.load(file, cls=RuntimeDecoder)
counts_hw = retrieved_job.get_counts()

# The real hardware used different qubits than the simulator
counts_hw = marginal_counts(counts_hw,[1,2,3,5])
FS = 50


# Plot the comparison
probs_sim = counts_to_probs(counts_sim)
probs_hw = counts_to_probs(counts_hw)
fig = plt.figure(figsize=(15,10))
# ax = fig.add_subplot()
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_alpha(0.3)
plt.bar(np.arange(len(probs_sim)),probs_sim*100,width=0.5,label='Sim',)
plt.bar(np.arange(len(probs_hw)),probs_hw*100,width=0.25,label='HW')
plt.xlabel('Measured value',fontsize=FS)
plt.ylabel('Probability (%)',fontsize=FS)
# plt.legend(loc='upper right',fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.ylim((0,28))
plt.legend()
plt.tight_layout()



# tikz_save('ketPsi.tikz')

# tikz_save('fig.tikz',
#             figureheight = '\\figureheight',
#             figurewidth = '\\figurewidth')

#%%
# Create the circuit for the PTDF matrix
mat_circuit, rootsum = matrix_circuit(exp_matrix, vqc.qregs) # scaled to set all singluar values to 1

# Get the number of qubits in each circuit
nv = vqc.num_qubits
nm = mat_circuit.num_qubits

# Build the combined circuit
circuit = copy.deepcopy(vqc)
circuit.barrier()
circuit.append(mat_circuit, list(np.arange(nm)))
circuit.barrier()

##
# Run on the statevector simulator
backend_sv = Aer.StatevectorSimulator()
t_circ_sv = transpile(circuit, backend_sv)

job_sv = backend_sv.run(t_circ_sv)
result_sv = job_sv.result()

# Get the state vector from the simulation
sv = result_sv.get_statevector(circuit, decimals=10)

outputstate = np.real(sv)

# get the indices of the result in the state vector
out = outputstate[:len(rootsum)]*vec_norm*rootsum
print('statevec result',out)
print('statevec mean', out@outp_range)
print('exact mean', exact_mu)


#%% Run the power flow circuit on real and simulated QC (Second step)
t_circ_sim = transpile(circuit, backend_sim)
t_circ_sim.measure_all()

job_sim = backend_sim.run(t_circ_sim,shots=shots)

result_sim = job_sim.result()
counts_sim = result_sim.get_counts()

# # Run on a real hardware
# t_circ_hw = transpile(circuit, backend_hw)
# t_circ_hw.measure_all()

# job_hw = backend_hw.run(t_circ_hw,shots=shots)
# job_monitor(job_hw)
# result_hw = job_hw.result()
# counts_hw = result_hw.get_counts()

# Retrieve the result from the real hardware

# retrieved_job = service.job("ciht3gf985671v615is0")
# with open("retrieved_job_2.json", "w") as file:
#     json.dump(retrieved_job.result(), file, cls=RuntimeEncoder)
    
with open("retrieved_job_2.json", "r") as file:
    retrieved_job = json.load(file, cls=RuntimeDecoder)
counts_hw = retrieved_job.get_counts()

FS = 50

# Plot the comparison
probs_sim = counts_to_probs(counts_sim)
probs_hw = counts_to_probs(counts_hw)
plt.figure(figsize=(15,10))
plt.bar(np.arange(len(probs_sim)),probs_sim*100,width=0.5,label='Sim',)
plt.bar(np.arange(len(probs_hw)),probs_hw*100,width=0.25,label='HW')
plt.xlabel('Measured value',fontsize=FS)
plt.ylabel('Probability (%)',fontsize=FS)
plt.legend(loc='upper right',fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.ylim((0,28))
plt.tight_layout()

# tikz_save('ketL.tikz')

# Calculate the estimated mean of the result
qres = np.sqrt(probs_sim)[:len(rootsum)]*vec_norm*rootsum
q_mu = np.dot(outp_range, qres[:len(res)])

#%% Do a classical simulation

####
# Create vectors with power injections
Nbin = 10**ndec # number of values in each vector

Plist = []
for P1 in Problist:
    P1_ = np.empty(0)
    for i, v in enumerate(P1):
        P1_ = np.concatenate((P1_,np.ones(int(v*Nbin))*i))
    if len(P1_) < Nbin:
        k = Nbin-len(P1_)
        P1_ = np.concatenate((P1_,np.ones(k)*np.max(P1_)))
    
    Plist.append(P1_)

N = 2003 # number of times to shuffle the vectors
rng = np.random.default_rng(1234)
l = 10 # number of random values taken each iteration
vals = []

for n in range(N):
    P = []
    for p in Plist: # Take l random values from each distribution
        P.append(rng.choice(p,l))
        
    Ps = abs(M@np.array(P)) # Compute the resulting line loading

    vals.extend(list(Ps))
    
mu = np.mean(vals) # 
sigma = np.std(vals)
me = 1.96*sigma/np.sqrt(N*l)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
conf_int_c = np.array([mu-me,mu+me])
cres = np.zeros(len(outp_range))
for i,v in enumerate(outp_range):
    cres[i] = np.sum(np.round(vals,2)==v)/(N*l)
    
# Plot quantum vs classical distribution
plt.figure()
plt.bar(outp_range,cres*100,label='cc',width=1*3*2)

plt.ylim((0,max(cres*100)*1.2))
plt.bar(outp_range,qres*100,label='qc',width=0.75*3*2)
plt.ylabel('Probability of flow in the line %',fontsize=15)
plt.xlabel('Line loading %',fontsize=15)
plt.legend()

print("Exact mean value:    \t%.4f" % exact_mu)
print("C MC mean value:\t%.4f" % mu)
print("Q MC mean value:\t%.4f" % q_mu)


#%%
# Calculation of the mean

weights = outp_range*rootsum
weights = np.pad(weights,(0,2**circuit.num_qubits-len(weights)),constant_values=0)

scaling = np.linalg.norm(weights)*vec_norm # scaling factor from normalizing the vectors

# create a unitary to compute the scaled mean of the distribution
urs = vec_to_unitary(weights, -1) # -1 so the result is stored in the last element

# Add the matrix to the circuit
um = copy.deepcopy(circuit)
um.unitary(urs,list(np.arange(circuit.num_qubits)),label='Estimate')
um.barrier()

# set target precision and confidence level for AQE
epsilon = 0.01
alpha = 0.05

# Define the estimation problem
problem = EstimationProblem(
    state_preparation=um,
    objective_qubits=list(np.arange(um.num_qubits))
)

# Define the IAE object
ae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100})
)

# Run the amplitude estimation
result = ae.estimate(problem)

conf_int = np.sqrt(np.array(result.confidence_interval_processed))*scaling
print("Exact mean value:    \t%.4f" % exact_mu)
print("Estimated mean value:\t%.4f" % (np.sqrt(result.estimation_processed)*scaling))
print("Confidence interval: \t[%.4f, %.4f]" % tuple(conf_int))

#%% Run the mean estimation circuit on real and simulated QC (Third step)

t_circ_sim = transpile(um, backend_sim)
t_circ_sim.measure_all()

job_sim = backend_sim.run(t_circ_sim,shots=shots)

result_sim = job_sim.result()
counts_sim = result_sim.get_counts()

# # Run on a real hardware
# t_circ_hw = transpile(um, basis_gates=basis_gates)
# t_circ_hw.draw('mpl').savefig('3bus.pdf')
# t_circ_hw.measure_all()

# job_hw = backend_hw.run(t_circ_hw,shots=shots)
# job_monitor(job_hw)
# result_hw = job_hw.result()
# counts_hw = result_hw.get_counts()

# Retrieve the result from the real hardware

# retrieved_job = service.job("cjsnvomiel5ovfe1suug")

# with open("retrieved_job_3.json", "w") as file:
#     json.dump(retrieved_job.result(), file, cls=RuntimeEncoder)
    
with open("retrieved_job_3.json", "r") as file:
    retrieved_job = json.load(file, cls=RuntimeDecoder)

counts_hw = retrieved_job.get_counts()

FS = 50

# Plot the comparison
probs_sim = counts_to_probs(counts_sim)
probs_hw = counts_to_probs(counts_hw)
plt.figure(figsize=(15,10))
plt.bar(np.arange(len(probs_sim)),probs_sim*100,width=0.5,label='Sim',)
plt.bar(np.arange(len(probs_hw)),probs_hw*100,width=0.25,label='HW')
plt.xlabel('Measured value',fontsize=FS)
plt.ylabel('Probability (%)',fontsize=FS)
plt.legend(loc='upper right',fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.ylim((0,28))
plt.tight_layout()

# tikz_save('ketV.tikz')

print('sim',probs_sim[-1],np.sqrt(probs_sim[-1])*scaling)
print('hw',probs_hw[-1],np.sqrt(probs_hw[-1])*scaling)

#%% Check the result using the statevector simulator

t_circ = transpile(um, backend_sv, optimization_level=1)

job = backend_sv.run(t_circ)
result = job.result()

# Get the state vector from the simulation
sv = result.get_statevector(um, decimals=10)
outputstate = np.real(sv)

# get the indices of the result in the state vector
out = outputstate[:]
print('statevec result',out[-1]*scaling)
print('exact mean', exact_mu)


#%% Estimate the probability of the line loading exceeding 90%

if max(outp_range) < 90: # If the output range (max injection times the PTDFs) doesn't reach 90%
    print('Output range is below 90pct')
else:
    outp_range2 = (outp_range>=90)
    
    weights = outp_range2*rootsum
    weights = np.pad(weights,(0,2**circuit.num_qubits-len(weights)),constant_values=0)
    
    scaling = np.linalg.norm(weights)*vec_norm # scaling factor from normalizing the vectors
    
    # create a unitary to compute the scaled mean of the distribution
    urs = vec_to_unitary(weights,-1)
    
    u90 = copy.deepcopy(circuit)
    u90.unitary(urs,list(np.arange(circuit.num_qubits)),label='Estimate')
    
    # Define the estimation problem
    problem = EstimationProblem(
        state_preparation=u90,
        objective_qubits=list(np.arange(u90.num_qubits))
    )
    
    # Run the amplitude estimation
    result = ae.estimate(problem)
    
    conf_int = np.sqrt(np.array(result.confidence_interval_processed))*scaling
    print("Exact probability of line loading over 90pct:    \t%.4f" % (sum(res[np.flatnonzero(outp_range2)])*100))
    print("Estimated probability of line loading over 90pct:\t%.4f" % (np.sqrt(result.estimation_processed)*scaling*100))
    print("Confidence interval: \t[%.4f, %.4f]" % tuple(conf_int*100))


    N = 2003 # number of times to shuffle the vectors
    l = 10 # number of random values taken each iteration
    vals = []

    for n in range(N):
        P = []
        for p in Plist: # Take l random values from each distribution
            P.append(rng.choice(p,l))
            
        Ps = abs(M@np.array(P)) # Compute the resulting line loading

        vals.extend(list((Ps>90)*1))
        
    mu = np.mean(vals)*100 
    sigma = np.std(vals)*100
    me = 1.96*sigma/np.sqrt(N*l)
