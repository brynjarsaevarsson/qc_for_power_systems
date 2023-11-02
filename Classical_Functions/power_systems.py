# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:15:05 2022

@author: brysa
"""

import numpy as np

bus_inds = {'Nr':0,'Type':1,'Pld':2,'Qld':3,'Gsh':4,'Bsh':5,'Vm':6,'Va':7}
br_inds = {'From':0,'To':1,'R':2,'X':3,'G':4,'B':5,'Tap':6,'Phase':7}

bus = {'Nr':0,'Type':0,'Pld':0,'Qld':0,'Gsh':0,'Bsh':0,'Vm':0,'Va':0}
branch = {'From':0,'To':0,'R':0,'X':0,'G':0,'B':0,'Tap':0,'Phase':0,'rating':0}
gen = {'bus':0, 'P':0, 'Q':0, 'V':1, 'MVA':100, 'Pmin':0, 'Pmax':0, 'Qmin':0, 'Qmax':0}

def make_Bmat(bus, branch):

    n_bus = len(bus['Nr'])
    n_br = len(branch['From'])

    Ys = 1/(1j*branch['X'])
    # Yp = branch['G']+  1j*branch['B']

    Yff = Ys
    Yft = - Ys
    Ytf = - Ys

    f = branch['From'].astype(int)
    t = branch['To'].astype(int)

    Ybus = np.zeros((n_bus,n_bus),dtype=complex)

    for i in range(n_br):
        Ybus[f[i],f[i]] += Yff[i]
        Ybus[f[i],t[i]] += Yft[i]
        Ybus[t[i],f[i]] += Ytf[i]
        Ybus[t[i],t[i]] += Yff[i]

    buscode = bus['Type']

    pq_index = np.where(buscode == 1)[0] # Find indices for all PQ-busses
    pv_index = np.where(buscode == 2)[0] # Find indices for all PV-busses

    dQ = pq_index
    dV = pq_index

    dP = np.concatenate((pv_index, pq_index))
    dTheta = np.concatenate((pv_index, pq_index))

    Bp = np.imag(Ybus[dP, :][:, dTheta])
    Bpp = np.imag(Ybus[dQ, :][:, dV])

    return Bp, Bpp

def make_Bdc(bus, branch):

    n_bus = len(bus['Nr'])
    n_br = len(branch['From'])

    b = 1/(branch['X'])
    tap = np.ones(n_br)
    i = np.flatnonzero(branch['Tap'])
    tap[i] = branch['Tap'][i]
    b = b / tap

    f = branch['From'].astype(int)
    t = branch['To'].astype(int)

    Cft = np.zeros((n_br,n_bus))
    Cft[np.arange(n_br),f] = 1
    Cft[np.arange(n_br),t] = -1

    Bf = np.zeros((n_br,n_bus))
    Bf[np.arange(n_br),f] = b
    Bf[np.arange(n_br),t] = -b

    Bbus = Cft.T@ Bf

    Pfinj = b * (-branch['Phase'] * np.pi/180)

    Pbusinj = Cft.T @ Pfinj

    return Bbus, Bf, Pbusinj, Pfinj

def make_Ybus(bus, branch):

    n_bus = len(bus['Nr'])
    n_br = len(branch['From'])

    Ys = 1/(branch['R'] + 1j*branch['X'])
    Yp = branch['G']+  1j*branch['B']

    Yff = Ys + Yp/2
    Yft = - Ys
    Ytf = - Ys

    Ysh = (bus['Gsh'] + 1j*bus['Bsh'])

    f = branch['From'].astype(int)
    t = branch['To'].astype(int)

    Ybus = np.zeros((n_bus,n_bus),dtype=complex)

    for i in range(n_br):
        at = branch['Tap'][i]
        if at > 0:
            Ybus[f[i],f[i]] += Yff[i]/at+Yff[i]*(1-at)/(at**2)
            Ybus[f[i],t[i]] += Yft[i]/at
            Ybus[t[i],f[i]] += Ytf[i]/at
            Ybus[t[i],t[i]] += Yff[i]/at+Yff[i]*(at-1)/at
        else:
            Ybus[f[i],f[i]] += Yff[i]
            Ybus[f[i],t[i]] += Yft[i]
            Ybus[t[i],f[i]] += Ytf[i]
            Ybus[t[i],t[i]] += Yff[i]

    for i in range(n_bus):
        Ybus[i,i] += Ysh[i]

    Sbus = bus['Pld']+1j*bus['Qld']

    V0 = bus['Vm']*np.exp(1j*np.deg2rad(bus['Va']))

    buscode = bus['Type']

    pq_index = np.where(buscode == 1)[0] # Find indices for all PQ-busses
    pv_index = np.where(buscode == 2)[0] # Find indices for all PV-busses
    ref = np.where(buscode == 3)[0] # Find index for ref bus

    # Create Branch Matrices
    # Create the two branch admittance matrices
    Y_from = np.zeros((n_br,n_bus),dtype=complex)
    Y_to = np.zeros((n_br,n_bus),dtype=complex)

    for k in range(0,len(f)): # Fill in the matrices
        Y_from[k,f[k]] = Ys[k]
        Y_from[k,t[k]] = -Ys[k]
        Y_to[k,f[k]] = -Ys[k]
        Y_to[k,t[k]] = Ys[k]

    return Ybus, V0, Sbus, pq_index, pv_index, ref, Y_from, Y_to

def init_dicts(n_bus, n_br, n_gen):

    for l in bus.keys():
        bus[l] = np.zeros(n_bus)

    for g in gen.keys():
        gen[g] = np.zeros(n_gen)

    for l in branch.keys():
        branch[l] = np.zeros(n_br)


def ptdf(bus, branch, full=True):
        
    # Get slack bus indices
    ref = np.flatnonzero(bus['Type']==3)

    n_bus = len(bus['Nr'])
    n_br = len(branch['From'])
    
    # Create the power flow matrix for mapping bus injections to line flows
    b = 1/(branch['X'])
    
    f = branch['From'].astype(int)
    t = branch['To'].astype(int)
    
    Cft = np.zeros((n_br,n_bus))
    Cft[np.arange(n_br),f] = 1
    Cft[np.arange(n_br),t] = -1
    
    Bf = np.zeros((n_br,n_bus))
    Bf[np.arange(n_br),f] = b
    Bf[np.arange(n_br),t] = -b
    
    # Bbus_full = Cft.T@Bf
    
    Bf_noref = np.delete(Bf,ref,axis=1)
    CFt_noref_T = np.delete(Cft,ref,axis=1).T
    
    PTDF = Bf_noref@np.linalg.inv(CFt_noref_T@Bf_noref) # Power Transfer Distribution Factors
    # The slack bus is not included in the PTDF matrix
    
    if full: # include the slack bus
        PTDF = np.insert(PTDF, ref , 0, axis=1)

    return PTDF

def lodf(branch, PTDF):
    
    nl, nb = PTDF.shape
    f = branch['From'].astype(int)
    t = branch['To'].astype(int)

    Cft = np.zeros((nl,nb))
    Cft[np.arange(nl),f] = 1
    Cft[np.arange(nl),t] = -1
    
    H = PTDF @ Cft.T
    h = np.expand_dims(np.diag(H),1)
    # LODF = H / (np.ones((nl, nl)) - np.ones((nl, 1)) @ h.T)
    LODF = np.divide(H,(np.ones((nl, nl)) - np.ones((nl, 1)) @ h.T),
                     out=np.zeros_like(H),
                     where=(np.ones((nl, nl)) - np.ones((nl, 1)) @ h.T)!=0)
    np.fill_diagonal(LODF, -1)
    
    return LODF


def IEEE_5bus(rm_line=None):

    n_bus = 5
    Sbase = 100

    bus_type = np.array([3, 1, 1, 1, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.06, 1.00, 1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([0.0, 40.0, 0.0, 0.0, 0.0])/Sbase
    Qgen = np.array([0.0, 30.0, 0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 20.0, 45.0, 40.0, 60.0])/Sbase
    Qld = np.array([0.0, 10.0, 15.0, 5.0, 10.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.02, 0.08, 0.06, 0.06, 0.04, 0.01, 0.08])
    X_br = np.array([0.06, 0.24, 0.25, 0.18, 0.12, 0.03, 0.24])
    G_br = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    B_br = np.array([0.03, 0.025, 0.02, 0.02, 0.015, 0.01, 0.025])*2

    br_f = np.array([1, 1, 2, 2, 2, 3, 4])-1
    br_t = np.array([2, 3, 3, 4, 5, 4, 5])-1

    ###
    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,0.104/X_br) # TODO add propper line rating
    # # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_5bus_qpf(rm_line=None):

    n_bus = 5
    # n_br = 7
    Sbase = 100

    bus_type = np.array([3, 1, 1, 1, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.05, 1.00, 1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([0.0, 20.0, 0.0, 0.0, 0.0])/Sbase
    Qgen = np.array([0.0, 5.0, 0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 10.0, 15.0, 10.0, 15.0])/Sbase
    Qld = np.array([0.0, 5.0, 7.5, 2.5, 5.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.02, 0.08, 0.06, 0.06, 0.04, 0.01, 0.08])*0
    X_br = np.array([ 1.03424425,  0.52637178,  0.90166637,  1.17900178, 29.20287936, 15.59879165,  0.67128607])
    G_br = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    B_br = np.array([0.03, 0.025, 0.02, 0.02, 0.015, 0.01, 0.025])*2*0

    br_f = np.array([1, 1, 1, 1, 2, 3, 4])-1
    br_t = np.array([2, 3, 4, 5, 3, 4, 5])-1

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.array([0.1, 0.2, 0.1, 0.1, 0.01, 0.01, 0.07]) # XXX Randomly chosen
    # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_5bus_sqpf(rm_line=None):

    n_bus = 5
    Sbase = 1

    bus_type = np.array([3, 2, 1, 2, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.00, 1.00, 1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
   
    Pgen = np.array([0.0, 2.0, 0.0, 1.0, 0.0])/Sbase
    Qgen = np.array([0.0, 0.0, 0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 0.0, 2.0, 0.0, 2.0])/Sbase
    Qld = np.array([0.0, 0.0, 0.0, 0.0, 0.0])/Sbase
    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_br = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])/100
    G_br = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    B_br = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    br_f = np.array([1, 1, 1, 2, 3, 4])-1
    br_t = np.array([2, 4, 5, 3, 4, 5])-1

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(len(br_f),1.0) # XXX Randomly chosen

    return bus, branch, gen

def IEEE_3bus(rm_line=None):

    n_bus = 3
    # n_br = 7
    Sbase = 100

    bus_type = np.array([3, 1, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([20.0, 60.0, 0.0,])/Sbase
    Qgen = np.array([0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 0.0, 80.0])/Sbase
    Qld = np.array([0.0, 0.0, 0.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.00, 0.00, 0.00])
    X_br = np.array([0.0125, 0.0125, 0.05])
    G_br = np.array([0.00, 0.00, 0.00])
    B_br = np.array([0.00, 0.00, 0.00])

    br_f = np.array([1, 1, 2])-1
    br_t = np.array([2, 3, 3])-1

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_3bus_qpf(rm_line=None):

    n_bus = 3
    # n_br = 7
    Sbase = 100

    bus_type = np.array([3, 1, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.03, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([5.0, 10.0, 0.0,])/Sbase
    Qgen = np.array([0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 0.0, 15.0])/Sbase
    Qld = np.array([0.0, 0.0, 5.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.00, 0.00, 0.00])
    X_br = np.array([1, 1, 2])
    G_br = np.array([0.00, 0.00, 0.00])
    B_br = np.array([0.00, 0.00, 0.00])

    br_f = np.array([1, 1, 2])-1
    br_t = np.array([2, 3, 3])-1
    
    line_ratings = np.array([0.05,  0.09,  0.06]) # XXX Randomly chosen

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)
        
        line_ratings = np.delete(line_ratings, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = line_ratings 
    # branch['Tap'] = tr_ratio

    return bus, branch, gen


def IEEE_3bus_sqpf(rm_line=None):

    n_bus = 3
    Sbase = 1

    bus_type = np.array([2, 2, 3])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([1.0, 2.0, 0.0,])/Sbase
    Qgen = np.array([0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 0.0, 0.0])/Sbase
    Qld = np.array([0.0, 0.0, 0.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.00, 0.00, 0.00])
    X_br = np.array([1, 1, 2])/100
    G_br = np.array([0.00, 0.00, 0.00])
    B_br = np.array([0.00, 0.00, 0.00])

    br_f = np.array([1, 1, 2])-1
    br_t = np.array([2, 3, 3])-1
    
    line_ratings = np.array([1,  2,  1.5]) # XXX Randomly chosen

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)
        
        line_ratings = np.delete(line_ratings, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = line_ratings 
    # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_9bus(rm_line=None):

    n_bus = 9

    Sbase = 100

    bus_type = np.array([3,2,2,1,1,1,1,1,1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([0.0, 163.0, 85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])/Sbase
    Qgen = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 100.0, 0.0, 125.0])/Sbase
    Qld = np.array([0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 35.0, 0.0, 50.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.0 , 0.017 , 0.039 , 0.0 , 0.0119 , 0.0085, 0.0, 0.032, 0.01])
    
    X_br = np.array([0.0576, 0.092, 0.17, 0.0586, 0.1008, 0.072, 0.0625, 0.161, 0.085])
    G_br = np.zeros(9)
    B_br = np.array([0.0, 0.158, 0.358, 0.0, 0.209 , 0.149, 0.0, 0.306, 0.176])

    br_f = np.array([1, 4, 5, 3, 6, 7, 8, 8, 9])-1
    br_t = np.array([4, 5, 6, 6, 7, 8, 2, 9, 4])-1
    

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,1/(X_br*5)) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_14bus(rm_line=None):

    n_bus = 14

    Sbase = 100

    bus_type = np.array([3,2,2,1,1,2,1,2,1,1,1,1,1,1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.06, 1.045, 1.01, 1.00, 1.00, 1.07, 1.00, 1.09, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    Va = np.zeros(n_bus)
    Pgen = np.array([232.4, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])/Sbase
    Qgen = np.array([-16.9, 42.4, 23.4, 0.0, 0.0, 12.2, 0.0, 17.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])/Sbase

    Pld = np.array([0.0, 21.7, 94.2, 47.8, 7.6, 11.2, 0.0, 0.0, 29.5, 9.0, 3.5, 6.1, 13.5, 14.9])/Sbase
    Qld = np.array([0.0, 12.7, 19.0, 3.9, 1.6, 7.5, 0.0, 0.0, 16.6, 5.8, 1.8, 1.6, 5.8, 5.0])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    Bsh[8] = 0.19


    R_br = np.array([0.01938 , 0.04699 , 0.05811 , 0.05403 , 0.05695 , 0.060701,
                    0.01335 , 0.      , 0.      , 0.      , 0.      , 0.      ,
                    0.03181 , 0.09498 , 0.12291 , 0.06615 , 0.12711 , 0.8205  ,
                    0.22092 , 0.17093 ])
    X_br = np.array([0.05917, 0.19797, 0.17632, 0.22304, 0.17388, 0.17103, 0.04211,
                    0.25202, 0.20912, 0.17615, 0.55618, 0.11001, 0.0845 , 0.1989 ,
                    0.25581, 0.13027, 0.27038, 0.19207, 0.19988, 0.34802])
    G_br = np.zeros(20)
    B_br = np.array([0.0528, 0.0438, 0.0374, 0.0492, 0.034 , 0.0346, 0.0128, 0.    ,
                    0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                    0.    , 0.    , 0.    , 0.    ])

    br_f = np.array([1, 2, 2, 1, 2, 3, 4, 5, 4, 7, 4, 7, 9, 6, 6, 6, 7, 10, 12, 13])-1
    br_t = np.array([2, 3, 4, 5, 5, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 11, 13, 14])-1

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,1/(X_br*5)) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen


def IEEE_30bus(rm_line=None):

    n_bus = 30

    Sbase = 100

    bus_type = np.array([3, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                         ])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.0500, 1.0338, 1.0313, 1.0263, 1.0058, 1.0208, 1.0069,
                   1.0230, 1.0332, 1.0183, 1.0913, 1.0399, 1.0883, 1.0236,
                   1.0179, 1.0235, 1.0144, 1.0057, 1.0017, 1.0051, 1.0061,
                   1.0069, 1.0053, 0.9971, 1.0086, 0.9908, 1.0245, 1.0156,
                   1.0047, 0.9932
                   ])
    Vm[np.where(bus_type==1)] = 1
    Va = np.zeros(n_bus)
    Pgen = np.array([138.48, 57.56, 0.0, 0.0, 24.56, 0.0, 0.0, 35.0, 0.0, 0.0,
                     17.93, 0.0, 16.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                     ])/Sbase
    Qgen = np.array([-2.79, 2.47, 0.0, 0.0, 22.57, 0.0, 0.0, 34.84, 0.0, 0.0,
                     30.78, 0.0, 37.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])/Sbase
    Pld = np.array([0.0, 21.7, 2.4, 7.6, 94.2, 0.0, 22.8, 30.0, 0.0, 5.8, 0.0,
                    11.2, 0.0, 6.2, 8.2, 3.5, 9.0, 3.2, 9.5, 2.2, 17.5, 0.0,
                    3.2, 8.7, 0.0, 3.5, 0.0, 0.0, 2.4, 10.6
                    ])/Sbase
    Qld = np.array([0.0, 12.7, 1.2, 1.6, 19.0, 0.0, 10.9, 30.0, 0.0, 2.0, 0.0,
                    7.5, 0.0, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 0.0,
                    1.6, 6.7, 0.0, 2.3, 0.0, 0.0, 0.9, 1.9
                    ])/Sbase

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)

    # bus = np.c_[bus_nr,bus_type,Pgen-Pld,Qgen-Qld,Gsh,Bsh,Vm,Va]

    R_br = np.array([0.0192, 0.0452, 0.0570, 0.0132, 0.0472, 0.0581, 0.0119,
                     0.0460, 0.0267, 0.0120, 0.0, 0.0, 0.0, 0.0, 0.1231,
                     0.0662, 0.0945, 0.2210, 0.0524, 0.1073, 0.0639, 0.0340,
                     0.0936, 0.0324, 0.0348, 0.0727, 0.0116, 0.1000, 0.1150,
                     0.1320, 0.1885, 0.1093, 0.0, 0.2198, 0.3202, 0.2399,
                     0.0636, 0.0169, 0.0, 0.0, 0.2544
                     ])

    X_br = np.array([0.0575, 0.1652, 0.1737, 0.0379, 0.1983, 0.1763, 0.0414,
                     0.1160, 0.0820, 0.0420, 0.2080, 0.5560, 0.1100, 0.2560,
                     0.2559, 0.1304, 0.1987, 0.1997, 0.1923, 0.2185, 0.1292,
                     0.0680, 0.2090, 0.0845, 0.0749, 0.1499, 0.0236, 0.2020,
                     0.1790, 0.2700, 0.3292, 0.2087, 0.3960, 0.4153, 0.6027,
                     0.4533, 0.2000, 0.0599, 0.2080, 0.1400, 0.3800
                     ])
    G_br = np.zeros(len(X_br))

    B_br = np.array([0.0528, 0.0408, 0.0368, 0.0084, 0.0418, 0.0374, 0.0090,
                     0.0204, 0.0170, 0.0090, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4028, 0.0130,
                     0.0, 0.0, 0.0
                     ])*2


    br_f = np.array([1, 1, 2, 3, 2, 2, 4, 5, 6, 6, 6, 6, 9, 4, 12, 12, 12, 14,
                     16, 15, 18, 19, 10, 10, 10, 10, 21, 15, 22, 23, 24, 25,
                     28, 27, 27, 29, 8, 6, 9, 12, 25
                     ])-1
    br_t = np.array([2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 12, 14, 15, 16,
                     15, 17, 18, 19, 20, 20, 17, 21, 22, 22, 23, 24, 24, 25,
                     27, 27, 29, 30, 30, 28, 28, 11, 13, 26
                     ])-1

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)


    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen



def IEEE_118bus(rm_line=None):

    n_bus = 118

    Sbase = 100

    bus_type = np.array([2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2,
                         2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2,
                         1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2,
                         2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 3, 2, 1, 2,
                         2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2,
                         2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1,
                         1, 2, 2, 2, 2, 1, 1, 2, 1, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([0.955, 0.971, 0.968, 0.998, 1.002, 0.99, 0.989, 1.015, 1.043,
                   1.05, 0.985, 0.99, 0.968, 0.984, 0.97, 0.984, 0.995, 0.973,
                   0.963, 0.958, 0.959, 0.97, 1, 0.992, 1.05, 1.015, 0.968,
                   0.962, 0.963, 0.968, 0.967, 0.964, 0.972, 0.986, 0.981, 0.98,
                   0.992, 0.962, 0.97, 0.97, 0.967, 0.985, 0.978, 0.985, 0.987,
                   1.005, 1.017, 1.021, 1.025, 1.001, 0.967, 0.957, 0.946, 0.955,
                   0.952, 0.954, 0.971, 0.959, 0.985, 0.993, 0.995, 0.998, 0.969,
                   0.984, 1.005, 1.05, 1.02, 1.003, 1.035, 0.984, 0.987, 0.98,
                   0.991, 0.958, 0.967, 0.943, 1.006, 1.003, 1.009, 1.04, 0.997,
                   0.989, 0.985, 0.98, 0.985, 0.987, 1.015, 0.987, 1.005, 0.985,
                   0.98, 0.993, 0.987, 0.991, 0.981, 0.993, 1.011, 1.024, 1.01,
                   1.017, 0.993, 0.991, 1.001, 0.971, 0.965, 0.962, 0.952,
                   0.967, 0.967, 0.973, 0.98, 0.975, 0.993, 0.96, 0.96, 1.005,
                   0.974, 0.949])
    Vm[np.where(bus_type==1)] = 1
    Va = np.zeros(n_bus)
    Pgen = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0.85, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 2.2, 3.14, 0, 0, 0, 0, 0.07, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19, 0, 0, 2.04, 0, 0, 0, 0,
                     0.48, 0, 0, 0, 0, 1.55, 0, 1.6, 0, 0, 0, 3.91, 3.92, 0, 0, 
                     5.164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.77, 0, 0, 0, 0, 0, 
                     0, 0.04, 0, 6.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.52, 0, 0,
                     0.4, 0, 0, 0, 0, 0, 0, 0, 0.36, 0, 0, 0, 0, 0, 0, 0
                     ])
    Qgen = np.zeros(n_bus)
    Pld = np.array([0.51, 0.2, 0.39, 0.39, 0, 0.52, 0.19, 0.28, 0, 0, 0.7, 0.47,
                    0.34, 0.14, 0.9, 0.25, 0.11, 0.6, 0.45, 0.18, 0.14, 0.1, 
                    0.07, 0.13, 0, 0, 0.71, 0.17, 0.24, 0, 0.43, 0.59, 0.23, 
                    0.59, 0.33, 0.31, 0, 0, 0.27, 0.66, 0.37, 0.96, 0.18, 0.16, 
                    0.53, 0.28, 0.34, 0.2, 0.87, 0.17, 0.17, 0.18, 0.23, 1.13, 
                    0.63, 0.84, 0.12, 0.12, 2.77, 0.78, 0, 0.77, 0, 0, 0, 0.39,
                    0.28, 0, 0, 0.66, 0, 0.12, 0.06, 0.68, 0.47, 0.68, 0.61, 
                    0.71, 0.39, 1.3, 0, 0.54, 0.2, 0.11, 0.24, 0.21, 0, 0.48, 0,
                    1.63, 0.1, 0.65, 0.12, 0.3, 0.42, 0.38, 0.15, 0.34, 0.42, 
                    0.37, 0.22, 0.05, 0.23, 0.38, 0.31, 0.43, 0.5, 0.02, 0.08, 
                    0.39, 0, 0.68, 0.06, 0.08, 0.22, 1.84, 0.2, 0.33
                    ])
    Qld = np.array([0.27, 0.09, 0.1, 0.12, 0, 0.22, 0.02, 0, 0, 0, 0.23, 0.1, 
                    0.16, 0.01, 0.3, 0.1, 0.03, 0.34, 0.25, 0.03, 0.08, 0.05, 
                    0.03, 0, 0, 0, 0.13, 0.07, 0.04, 0, 0.27, 0.23, 0.09, 0.26,
                    0.09, 0.17, 0, 0, 0.11, 0.23, 0.1, 0.23, 0.07, 0.08, 0.22, 
                    0.1, 0, 0.11, 0.3, 0.04, 0.08, 0.05, 0.11, 0.32, 0.22, 0.18,
                    0.03, 0.03, 1.13, 0.03, 0, 0.14, 0, 0, 0, 0.18, 0.07, 0, 0, 
                    0.2, 0, 0, 0, 0.27, 0.11, 0.36, 0.28, 0.26, 0.32, 0.26, 0, 
                    0.27, 0.1, 0.07, 0.15, 0.1, 0, 0.1, 0, 0.42, 0, 0.1, 0.07, 
                    0.16, 0.31, 0.15, 0.09, 0.08, 0, 0.18, 0.15, 0.03, 0.16, 
                    0.25, 0.26, 0.16, 0.12, 0.01, 0.03, 0.3, 0, 0.13, 0, 0.03,
                    0.07, 0, 0.08, 0.15
                    ])

    Gsh = np.zeros(n_bus)
    Bsh = np.zeros(n_bus)


    R_br = np.array([0.0303, 0.0129, 0.00176, 0.0241, 0.0119, 0.00459, 0.00244, 
                     0, 0.00258, 0.0209, 0.0203, 0.00595, 0.0187, 0.0484, 
                     0.00862, 0.02225, 0.0215, 0.0744, 0.0595, 0.0212, 0.0132, 
                     0.0454, 0.0123, 0.01119, 0.0252, 0.012, 0.0183, 0.0209, 
                     0.0342, 0.0135, 0.0156, 0, 0.0318, 0.01913, 0.0237, 0, 
                     0.00431, 0.00799, 0.0474, 0.0108, 0.0317, 0.0298, 0.0229,
                     0.038, 0.0752, 0.00224, 0.011, 0.0415, 0.00871, 0.00256, 
                     0, 0.0321, 0.0593, 0.00464, 0.0184, 0.0145, 0.0555, 0.041,
                     0.0608, 0.0413, 0.0224, 0.04, 0.038, 0.0601, 0.0191, 
                     0.0715, 0.0715, 0.0684, 0.0179, 0.0267, 0.0486, 0.0203,
                     0.0405, 0.0263, 0.073, 0.0869, 0.0169, 0.00275, 0.00488, 
                     0.0343, 0.0474, 0.0343, 0.0255, 0.0503, 0.0825, 0.0803, 
                     0.04739, 0.0317, 0.0328, 0.00264, 0.0123, 0.00824, 0, 
                     0.00172, 0, 0.00901, 0.00269, 0.018, 0.018, 0.0482, 0.0258,
                     0, 0.0224, 0.00138, 0.0844, 0.0985, 0, 0.03, 0.00221, 
                     0.00882, 0.0488, 0.0446, 0.00866, 0.0401, 0.0428, 0.0405, 
                     0.0123, 0.0444, 0.0309, 0.0601, 0.00376, 0.00546, 0.017, 
                     0.0294, 0.0156, 0.00175, 0, 0.0298, 0.0112, 0.0625, 0.043,
                     0.0302, 0.035, 0.02828, 0.02, 0.0239, 0.0139, 0.0518, 
                     0.0238, 0.0254, 0.0099, 0.0393, 0.0387, 0.0258, 0.0481, 
                     0.0223, 0.0132, 0.0356, 0.0162, 0.0269, 0.0183, 0.0238, 
                     0.0454, 0.0648, 0.0178, 0.0171, 0.0173, 0.0397, 0.018, 
                     0.0277, 0.0123, 0.0246, 0.016, 0.0451, 0.0466, 0.0535, 
                     0.0605, 0.00994, 0.014, 0.053, 0.0261, 0.053, 0.0105, 
                     0.03906, 0.0278, 0.022, 0.0247, 0.00913, 0.0615, 0.0135, 
                     0.0164, 0.0023, 0.00034, 0.0329, 0.0145, 0.0164
                     ])
    X_br = np.array([0.0999, 0.0424, 0.00798, 0.108, 0.054, 0.0208, 0.0305, 
                     0.0267, 0.0322, 0.0688, 0.0682, 0.0196, 0.0616, 0.16, 
                     0.034, 0.0731, 0.0707, 0.2444, 0.195, 0.0834, 0.0437, 
                     0.1801, 0.0505, 0.0493, 0.117, 0.0394, 0.0849, 0.097, 
                     0.159, 0.0492, 0.08, 0.0382, 0.163, 0.0855, 0.0943, 
                     0.0388, 0.0504, 0.086, 0.1563, 0.0331, 0.1153, 0.0985, 
                     0.0755, 0.1244, 0.247, 0.0102, 0.0497, 0.142, 0.0268, 
                     0.0094, 0.0375, 0.106, 0.168, 0.054, 0.0605, 0.0487, 
                     0.183, 0.135, 0.2454, 0.1681, 0.0901, 0.1356, 0.127,
                     0.189, 0.0625, 0.323, 0.323, 0.186, 0.0505, 0.0752, 
                     0.137, 0.0588, 0.1635, 0.122, 0.289, 0.291, 0.0707, 
                     0.00955, 0.0151, 0.0966, 0.134, 0.0966, 0.0719, 0.2293,
                     0.251, 0.239, 0.2158, 0.145, 0.15, 0.0135, 0.0561, 0.0376,
                     0.0386, 0.02, 0.0268, 0.0986, 0.0302, 0.0919, 0.0919, 
                     0.218, 0.117, 0.037, 0.1015, 0.016, 0.2778, 0.324, 0.037,
                     0.127, 0.4115, 0.0355, 0.196, 0.18, 0.0454, 0.1323, 0.141,
                     0.122, 0.0406, 0.148, 0.101, 0.1999, 0.0124, 0.0244, 
                     0.0485, 0.105, 0.0704, 0.0202, 0.037, 0.0853, 0.03665, 
                     0.132, 0.148, 0.0641, 0.123, 0.2074, 0.102, 0.173, 0.0712,
                     0.188, 0.0997, 0.0836, 0.0505, 0.1581, 0.1272, 0.0848, 
                     0.158, 0.0732, 0.0434, 0.182, 0.053, 0.0869, 0.0934, 0.108,
                     0.206, 0.295, 0.058, 0.0547, 0.0885, 0.179, 0.0813, 0.1262,
                     0.0559, 0.112, 0.0525, 0.204, 0.1584, 0.1625, 0.229, 
                     0.0378, 0.0547, 0.183, 0.0703, 0.183, 0.0288, 0.1813,
                     0.0762, 0.0755, 0.064, 0.0301, 0.203, 0.0612, 0.0741,
                     0.0104, 0.00405, 0.14, 0.0481, 0.0544
                     ])

    G_br = np.zeros(len(X_br))
    B_br = np.array([0.0127, 0.00541, 0.00105, 0.0142, 0.00713, 0.00275, 0.581,
                     0, 0.615, 0.00874, 0.00869, 0.00251, 0.00786, 0.0203, 
                     0.00437, 0.00938, 0.00908, 0.03134, 0.0251, 0.0107, 0.0222,
                     0.0233, 0.00649, 0.00571, 0.0149, 0.00505, 0.0108, 0.0123,
                     0.0202, 0.0249, 0.0432, 0, 0.0882, 0.0108, 0.0119, 0, 
                     0.257, 0.454, 0.01995, 0.00415, 0.05865, 0.01255, 0.00963,
                     0.01597, 0.0316, 0.00134, 0.00659, 0.0183, 0.00284, 
                     0.00492, 0, 0.0135, 0.021, 0.211, 0.00776, 0.00611, 0.0233,
                     0.0172, 0.03034, 0.02113, 0.0112, 0.0166, 0.0158, 0.0236,
                     0.00802, 0.043, 0.043, 0.0222, 0.00629, 0.00937, 0.0171,
                     0.00698, 0.02029, 0.0155, 0.0369, 0.0365, 0.0101, 0.00366,
                     0.00187, 0.0121, 0.0166, 0.0121, 0.00894, 0.0299, 0.02845,
                     0.0268, 0.02823, 0.0188, 0.0194, 0.00728, 0.00734, 0.0049,
                     0, 0.108, 0, 0.523, 0.19, 0.0124, 0.0124, 0.0289, 0.0155, 
                     0, 0.01341, 0.319, 0.03546, 0.0414, 0, 0.061, 0.05099, 
                     0.00439, 0.0244, 0.02222, 0.00589, 0.01684, 0.018, 0.062,
                     0.00517, 0.0184, 0.0519, 0.02489, 0.00632, 0.00324, 0.0236,
                     0.0114, 0.00935, 0.404, 0, 0.04087, 0.01898, 0.0129, 
                     0.0174, 0.00617, 0.0138, 0.02225, 0.0138, 0.0235, 0.00967,
                     0.0264, 0.053, 0.0107, 0.0274, 0.0207, 0.01634, 0.0109, 
                     0.0203, 0.00938, 0.00555, 0.0247, 0.0272, 0.0115, 0.0127,
                     0.0143, 0.0273, 0.0236, 0.0302, 0.00737, 0.012, 0.0238, 
                     0.0108, 0.0164, 0.00732, 0.0147, 0.0268, 0.02705, 0.02035,
                     0.0204, 0.031, 0.00493, 0.00717, 0.0236, 0.00922, 0.0236, 
                     0.0038, 0.02305, 0.0101, 0.01, 0.031, 0.00384, 0.0259, 
                     0.00814, 0.00986, 0.00138, 0.082, 0.0179, 0.00599, 0.00678
                     ])
    # X_br = np.ones(len(X_br))*0.1
    # R_br = np.ones(len(X_br))*0.01
    # B_br = np.ones(len(X_br))*0.0


    br_f = np.array([1, 1, 4, 3, 5, 6, 8, 8, 9, 4, 5, 11, 2, 3, 7, 11, 12, 13, 
                     14, 12, 15, 16, 17, 18, 19, 15, 20, 21, 22, 23, 23, 26, 
                     25, 27, 28, 30, 8, 26, 17, 29, 23, 31, 27, 15, 19, 35, 35,
                     33, 34, 34, 38, 37, 37, 30, 39, 40, 40, 41, 43, 34, 44, 
                     45, 46, 46, 47, 42, 42, 45, 48, 49, 49, 51, 52, 53, 49, 
                     49, 54, 54, 55, 56, 50, 56, 51, 54, 56, 56, 55, 59, 59, 
                     60, 60, 61, 63, 63, 64, 38, 64, 49, 49, 62, 62, 65, 66, 
                     65, 47, 49, 68, 69, 24, 70, 24, 71, 71, 70, 70, 69, 74, 
                     76, 69, 75, 77, 78, 77, 77, 79, 68, 81, 77, 82, 83, 83, 
                     84, 85, 86, 85, 85, 88, 89, 89, 90, 89, 89, 91, 92, 92, 
                     93, 94, 80, 82, 94, 80, 80, 80, 92, 94, 95, 96, 98, 99, 
                     100, 92, 101, 100, 100, 103, 103, 100, 104, 105, 105, 105,
                     106, 108, 103, 109, 110, 110, 17, 32, 32, 27, 114, 68, 12,
                     75, 76
                     ])-1
    br_t = np.array([2, 3, 5, 5, 6, 7, 9, 5, 10, 11, 11, 12, 12, 12, 12, 13, 
                     14, 15, 15, 16, 17, 17, 18, 19, 20, 19, 21, 22, 23, 24, 
                     25, 25, 27, 28, 29, 17, 30, 30, 31, 31, 32, 32, 32, 33, 
                     34, 36, 37, 37, 36, 37, 37, 39, 40, 38, 40, 41, 42, 42, 
                     44, 43, 45, 46, 47, 48, 49, 49, 49, 49, 49, 50, 51, 52, 
                     53, 54, 54, 54, 55, 56, 56, 57, 57, 58, 58, 59, 59, 59, 
                     59, 60, 61, 61, 62, 62, 59, 64, 61, 65, 65, 66, 66, 66, 
                     67, 66, 67, 68, 69, 69, 69, 70, 70, 71, 72, 72, 73, 74, 
                     75, 75, 75, 77, 77, 77, 78, 79, 80, 80, 80, 81, 80, 82, 
                     83, 84, 85, 85, 86, 87, 88, 89, 89, 90, 90, 91, 92, 92, 
                     92, 93, 94, 94, 95, 96, 96, 96, 97, 98, 99, 100, 100, 
                     96, 97, 100, 100, 101, 102, 102, 103, 104, 104, 105, 
                     106, 105, 106, 107, 108, 107, 109, 110, 110, 111, 112,
                     113, 113, 114, 115, 115, 116, 117, 118, 118
                     ])-1

    # tf = np.concatenate((br_f,br_t))
    # counts = np.bincount(np.sort(tf))
    # np.all(np.unique(np.sort(tf))==np.arange(len(np.unique(np.sort(tf)))))
    # print(max(counts))
    # print(np.argmax(counts)+1)
    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen

def IEEE_300bus(rm_line=None):

    n_bus = 300

    Sbase = 100

    bus_nrs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 33, 34, 35, 36,
                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
                        52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70,
                        71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 84, 85, 86, 87,
                        88, 89, 90, 91, 92, 94, 97, 98, 99, 100, 102, 103, 104,
                        105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117,
                        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                        129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                        140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
                        151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                        173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                        184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
                        206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
                        217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
                        228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
                        239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
                        250, 281, 319, 320, 322, 323, 324, 526, 528, 531, 552,
                        562, 609, 664, 1190, 1200, 1201, 2040, 7001, 7002, 7003,
                        7011, 7012, 7017, 7023, 7024, 7039, 7044, 7049, 7055,
                        7057, 7061, 7062, 7071, 7130, 7139, 7166, 9001, 9002,
                        9003, 9004, 9005, 9006, 9007, 9012, 9021, 9022, 9023,
                        9024, 9025, 9026, 9031, 9032, 9033, 9034, 9035, 9036,
                        9037, 9038, 9041, 9042, 9043, 9044, 9051, 9052, 9053,
                        9054, 9055, 9071, 9072, 9121, 9533
                        ])


    bus_type = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                         1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2,
                         1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1,
                         1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
                         2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
                         2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1
                         ])
    bus_nr = np.arange(n_bus)
    Vm = np.array([1.0284, 1.0354, 0.9971, 1.0308, 1.0191, 1.0312, 0.9934,
                   1.0153, 1.0034, 1.0205, 1.0057, 0.9974, 0.9977, 0.9991,
                   1.0343, 1.0315, 1.0649, 0.9820, 1.0010, 0.9752, 0.9963,
                   1.0501, 1.0057, 1.0234, 0.9986, 0.9750, 1.0244, 1.0414,
                   0.9757, 1.0011, 1.0201, 1.0202, 1.0535, 1.0216, 1.0292,
                   1.0448, 1.0006, 1.0086, 1.0215, 1.0344, 0.9777, 1.0019,
                   1.0475, 1.0253, 0.9979, 0.9959, 1.0050, 1.0150, 1.0335,
                   0.9918, 0.9789, 1.0246, 0.9906, 1.0160, 0.9583, 0.9480,
                   0.9630, 0.9513, 0.9793, 0.9696, 0.9775, 0.9964, 0.9632,
                   0.9837, 0.9900, 0.9820, 0.9872, 1.0340, 1.0250, 0.9872,
                   0.9909, 0.9921, 1.0151, 1.0317, 1.0272, 1.0520, 1.0520,
                   0.9930, 1.0183, 1.0000, 0.9894, 1.0060, 1.0008, 1.0288,
                   0.9958, 1.0223, 1.0095, 0.9900, 0.9749, 0.9730, 0.9725,
                   0.9700, 0.9747, 0.9603, 1.0249, 0.9348, 0.9298, 1.0435,
                   0.9584, 0.9871, 0.9728, 1.0006, 1.0233, 1.0103, 0.9978,
                   1.0001, 1.0024, 1.0028, 1.0191, 0.9861, 1.0045, 1.0020,
                   1.0220, 1.0193, 1.0476, 1.0471, 1.0550, 1.0117, 1.0430,
                   1.0510, 1.0155, 1.0435, 1.0160, 1.0081, 1.0528, 1.0528,
                   1.0577, 1.0735, 0.9869, 1.0048, 1.0535, 1.0435, 0.9663,
                   1.0177, 0.9630, 0.9845, 0.9987, 0.9867, 0.9998, 1.0360,
                   0.9918, 1.0410, 0.9839, 1.0002, 0.9973, 0.9715, 1.0024,
                   0.9879, 0.9290, 0.9829, 1.0244, 0.9837, 1.0622, 0.9730,
                   1.0522, 1.0077, 0.9397, 0.9699, 0.9793, 1.0518, 1.0447,
                   0.9717, 1.0386, 1.0522, 1.0650, 1.0650, 1.0533, 0.9975,
                   1.0551, 1.0435, 0.9374, 0.9897, 1.0489, 1.0357, 0.9695,
                   0.9907, 1.0150, 0.9528, 0.9550, 0.9692, 0.9908, 1.0033,
                   0.9718, 0.9838, 0.9992, 1.0137, 0.9929, 0.9999, 0.9788,
                   1.0017, 1.0132, 1.0100, 0.9919, 0.9866, 0.9751, 1.0215,
                   1.0075, 1.0554, 1.0080, 1.0000, 1.0500, 0.9965, 1.0002,
                   0.9453, 1.0180, 1.0000, 1.0423, 1.0496, 1.0400, 1.0535,
                   1.0414, 1.0000, 1.0387, 1.0095, 1.0165, 1.0558, 1.0100,
                   1.0000, 1.0237, 1.0500, 0.9930, 1.0100, 0.9921, 0.9711,
                   0.9651, 0.9688, 0.9760, 0.9752, 1.0196, 1.0251, 1.0152,
                   1.0146, 1.0005, 0.9810, 0.9750, 0.9429, 0.9723, 0.9604,
                   1.0009, 0.9777, 0.9583, 1.0309, 1.0128, 1.0244, 1.0122,
                   0.9653, 1.0507, 1.0507, 1.0323, 1.0145, 1.0507, 1.0507,
                   1.0507, 1.0290, 1.0500, 1.0145, 1.0507, 0.9967, 1.0212,
                   1.0145, 1.0017, 0.9893, 1.0507, 1.0507, 1.0145, 1.0117,
                   0.9945, 0.9833, 0.9768, 1.0117, 1.0029, 0.9913, 1.0023,
                   0.9887, 0.9648, 0.9747, 0.9706, 0.9649, 0.9657, 0.9318,
                   0.9441, 0.9286, 0.9973, 0.9506, 0.9598, 0.9570, 0.9391,
                   0.9636, 0.9501, 0.9646, 0.9790, 1.0000, 0.9786, 1.0000,
                   1.0000, 1.0000, 0.9752, 0.9803, 0.9799, 1.0402
                   ])

    Vm[np.where(bus_type==1)] = 1
    Va = np.zeros(n_bus)

    Pg = np.array([0, 0, 0, 0, 0, 375, 155, 290, 68, 117, 1930, 240, 0, 0, 281,
                   696, 84, 217, 103, 372, 216, 0, 205, 0, 228, 84, 200, 1200,
                   1200, 475, 1973, 424, 272, 100, 450, 250, 303, 345, 300, 600,
                   250, 550, 575.43, 170, 84, 467, 623, 1210, 234, 372, 330,
                   185, 410, 500, 37, 0, 45, 165, 400, 400, 116, 1292, 700, 553,
                   0, 0, 0, 50, 8
                   ])/Sbase
    Qg = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0
                   ])/Sbase


    gen_bus = np.logical_or(bus_type==2,bus_type==3)
    Pgen = np.zeros(n_bus)
    Pgen[np.where(gen_bus)] = Pg
    Qgen = np.zeros(n_bus)
    Qgen[np.where(gen_bus)] = Qg

    Pld = np.array([90, 56, 20, 0, 353, 120, 0, 63, 96, 153, 83, 0, 58, 160,
                    126.7, 0, 561, 0, 605, 77, 81, 21, 0, 45, 28, 69, 55, 0, 0,
                    0, 85, 155, 0, 46, 86, 0, 39, 195, 0, 0, 58, 41, 92, -5, 61,
                    69, 10, 22, 98, 14, 218, 0, 227, 0, 70, 0, 0, 56, 116, 57,
                    224, 0, 208, 74, 0, 48, 28, 0, 37, 0, 0, 0, 0, 44.2, 66,
                    17.4, 15.8, 60.3, 39.9, 66.7, 83.5, 0, 77.8, 32, 8.6, 49.6,
                    4.6, 112.1, 30.7, 63, 19.6, 26.2, 18.2, 0, 0, 0, 14.1, 0,
                    777, 535, 229.1, 78, 276.4, 514.8, 57.9, 380.8, 0, 0, 0, 0,
                    0, 0, 0, 169.2, 55.2, 273.6, 1019.2, 595, 387.7, 145, 56.5,
                    89.5, 0, 24, 0, 0, 63, 0, 0, 0, 17, 0, 70, 200, 75, 123.5,
                    0, 33, 0, 35, 85, 0, 0, 0, 0, 299.9, 0, 0, 481.8, 763.6,
                    26.5, 163.5, 0, 176, 5, 28, 427.4, 74, 69.5, 73.4, 240.7,
                    40, 136.8, 0, 59.8, 59.8, 182.6, 7, 0, 489, 800, 0, 0, 0,
                    10, 43, 64, 35, 27, 41, 38, 42, 72, 0, 12, -21, 7, 38, 0,
                    96, 0, 0, 22, 47, 176, 100, 131, 0, 285, 171, 328, 428, 173,
                    410, 0, 538, 223, 96, 0, 159, 448, 404, 572, 269, 0, 0, 255,
                    0, 0, 0, 0, 8, 0, 61, 77, 61, 29, 29, -23, -33.1, 115.8,
                    2.4, 2.4, -14.9, 24.7, 145.3, 28.1, 14, -11.1, 50.5, 29.6,
                    -113.7, 100.31, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    , 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.2, 2.71, 0.86, 0, 0, 0, 0,
                    4.75, 1.53, 0, 1.35, 0.45, 0.45, 1.84, 1.39, 1.89, 1.55,
                    1.66, 3.03, 1.86, 2.58, 1.01, 0.81, 1.6, 0, 35.81, 30,
                    26.48, 0, 0, 1.02, 1.02, 3.8, 1.19
                    ])/Sbase
    Qld = np.array([49, 15, 0, 0, 130, 41, 0, 14, 43, 33, 21, 0, 10, 60, 23, 0,
                    220, 0, 120, 1, 23, 7, 0, 12, 9, 13, 6, 0, 0, 0, 32, 18, 0,
                    -21, 0, 0, 9, 29, 0, 0, 11.8, 19, 26, 5, 28, 3, 1, 10, 20,
                    1, 106, 0, 110, 0, 30, 0, 0, 20, 38, 19, 71, 0, 107, 28, 0,
                    14, 7, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 650, 0, 215, 55, 11.8,
                    1.4, 59.3, 82.7, 5.1, 37, 0, 0, 0, 0, 0, 0, 0, 41.6, 18.2,
                    99.8, 135.2, 83.3, 114.7, 58, 24.5, 35.5, 0, 14, 0, 0, 25,
                    0, 0, 0, 9, 0, 5, 50, 50, -24.3, 0, 16.5, 0, 15, 24, 0.4, 0,
                    0, 0, 95.7, 0, 0, 205, 291.1, 0, 43, 0, 83, 4, 12, 173.6,
                    29, 49.3, 0, 89, 4, 16.6, 0, 24.3, 24.3, 43.6, 2, 0, 53, 72,
                    0, 0, 0, 3, 14, 21, 12, 12, 14, 13, 14, 24, -5, 2, -14.2, 2,
                    13, 0, 7, 0, 0, 16, 26, 105, 75, 96, 0, 100, 70, 188, 232,
                    99, 40, 0, 369, 148, 46, 0, 107, 143, 212, 244, 157, 0, 0,
                    149, 0, 0, 0, 0, 3, 0, 30, 33, 30, 14, 14, -17, -29.4, -24,
                    -12.6, -3.9, 26.5, -1.2, -34.9, -20.5, 2.5, -1.4, 17.4, 0.6,
                    76.7, 29.17, 34.17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.94, 0.28, 0, 0, 0, 0, 1.56,
                    0.53, 0, 0.47, 0.16, 0.16, 0.64, 0.48, 0.65, 0.54, 0.58, 1,
                    0.64, 0.89, 0.35, 0.28, 0.52, 0, 0, 23, 0, 0, 0, 0.35, 0.35,
                    1.25, 0.41
                    ])/Sbase

    Gsh = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0.14, 0, 0, 0, 0, 0, 0, 0.08, 0, 0.07,
                    0.02, 0.02, 0.1, 0.07, 0.1, 0.08, 0.09, 0, 0.1, 0.14, 0.05,
                    0.04, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.05, 0, 0.1
                    ])/Sbase
    Bsh = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 325, 0, 0, 55,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34.5, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, -212, 0, -103, 0, 0, 0, 0, 0, 0, 53, 0, 0, 0, 0,
                    0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -150, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -300, 0, 0, 0, 0,
                    0, 0, -150, 0, -140, 0, 0, 0, 0, 0, 0, 0, 45.6, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.4, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.72, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                   ])/Sbase



    R_br = np.array([6e-05, 0.0008, 0.02439, 0.03624, 0.01578, 0.01578,
                     0.01602, 0, 0, 0.05558, 0.11118, 0.11118, 0.07622,
                     0.07622, 0.0537, 1.1068, 0.44364, 0.50748, 0.66688,
                     0.6113, 0.4412, 0.30792, 0.0558, 0.73633, 0.76978,
                     0.75732, 0.07378, 0.03832, 0.36614, 1.0593, 0.1567,
                     0.13006, 0.54484, 0.15426, 0.3849, 0.4412, 0.23552,
                     0, 0.001, 0.001, 0.006, 0, 0.008, 0.001, 0.002, 0.006,
                     0.001, 0.001, 0.013, 0.013, 0.006, 0.008, 0.002, 0.006,
                     0.014, 0.065, 0.099, 0.096, 0.002, 0.002, 0.013, 0.016,
                     0.069, 0.004, 0.052, 0.019, 0.007, 0.036, 0.045, 0.043,
                     0, 0.0025, 0.006, 0.007, 0.001, 0.012, 0.006, 0.01, 0.004,
                     0.008, 0.022, 0.01, 0.017, 0.102, 0.047, 0.008, 0.032,
                     0.0006, 0.026, 0, 0.065, 0.031, 0.002, 0.026, 0.095,
                     0.013, 0.027, 0.028, 0.007, 0.009, 0.005, 0.052, 0.043,
                     0.025, 0.031, 0.037, 0.027, 0.025, 0.035, 0.065, 0.046,
                     0.159, 0.009, 0.002, 0.009, 0.016, 0.001, 0.0265, 0.051,
                     0.051, 0.032, 0.02, 0.036, 0.034, 0.018, 0.0256, 0.021,
                     0.018, 0.004, 0.0286, 0.016, 0.001, 0.014, 0.0891, 0.0782,
                     0.006, 0, 0.099, 0.022, 0.0035, 0.0035, 0.008, 0.012,
                     0.006, 0.047, 0.032, 0.1, 0.022, 0.019, 0.017, 0.278,
                     0.022, 0.038, 0.048, 0.024, 0.034, 0.053, 0.002, 0.045,
                     0.05, 0.016, 0.043, 0.019, 0.076, 0.044, 0.012, 0.157,
                     0.074, 0.07, 0.1, 0.109, 0.142, 0.017, 0.0036, 0.002,
                     0.0001, 0, 0, 0, 0.0022, 0, 0, 0.0808, 0.0965, 0.036,
                     0.0476, 0.0006, 0.0059, 0.0115, 0.0198, 0.005, 0.0077,
                     0.0165, 0.0059, 0.0049, 0.0059, 0.0078, 0.0026, 0.0076,
                     0.0021, 0.0016, 0.0017, 0.0079, 0.0078, 0.0017, 0.0026,
                     0.0021, 0.0002, 0.0043, 0.0039, 0.0091, 0.0125, 0.0056,
                     0.0015, 0.0005, 0.0007, 0.0005, 0.0562, 0.012, 0.0152,
                     0.0468, 0.043, 0.0489, 0.0013, 0.0291, 0.006, 0.0075,
                     0.0127, 0.0085, 0.0218, 0.0073, 0.0523, 0.1371, 0.0137,
                     0.0055, 0.1746, 0.0804, 0.011, 0.0008, 0.0029, 0.0066,
                     0.0024, 0.0018, 0.0044, 0.0002, 0.0018, 0.0669, 0.0558,
                     0.0807, 0.0739, 0.1799, 0.0904, 0.077, 0.0251, 0.0222,
                     0.0498, 0.0061, 0.0004, 0.0004, 0.0025, 0.0007, 0.0007,
                     0.0004, 0.033, 0.046, 0.0004, 0, 0.003, 0.002, 0.045,
                     0.048, 0.0031, 0.0024, 0.0031, 0.014, 0.03, 0.01, 0.015,
                     0.332, 0.009, 0.02, 0.034, 0.076, 0.04, 0.081, 0.124,
                     0.01, 0.046, 0.302, 0.073, 0.24, 0.0139, 0.0025, 0.0017,
                     0.0015, 0.0045, 0.004, 0, 0.0005, 0.0027, 0.0003, 0.0037,
                     0.001, 0.0016, 0.0003, 0.0014, 0.01, 0.0019, 0.001,
                     0.0005, 0.0009, 0.0019, 0.0026, 0.0013, 0, 0.0002, 0.0001,
                     0.0017, 0.0002, 0.0006, 0.0002, 0.0005, 0.0003, 0.0082,
                     0.0112, 0.0127, 0.0326, 0.0195, 0.0157, 0.036, 0.0268,
                     0.0428, 0.0351, 0.0616, 0, 0, 0, 0, 0, 0, 0, 0.0194,
                     0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0052,
                     0, 0.0005, 0, 0, 0.001, 0.0024, 0.0024, 0, 0.0013, 0.0005,
                     0.001, 0.0027, 0.0008, 0, 0.0012, 0.0013, 0.0009, 0.0003,
                     0, 0, 0.0008, 0, 0, 0, 0.02, 0.026, 0.003, 0.001, 0.0012,
                     0.001, 0.0005, 0.0005, 0.0001, 0.001, 0, 0.001, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                     ])
    X_br = np.array([0.00046, 0.00348, 0.43682, 0.64898, 0.37486, 0.37486,
                     0.38046, 0.152, 0.8, 0.24666, 0.49332, 0.49332, 0.43286,
                     0.43286, 0.07026, 0.95278, 2.8152, 3.2202, 3.944, 3.6152,
                     2.9668, 2.057, 0.24666, 4.6724, 4.8846, 4.8056, 0.06352,
                     0.02894, 2.456, 5.4536, 1.6994, 1.3912, 3.4572, 1.6729,
                     2.5712, 2.9668, 0.99036, 0.75, 0.006, 0.009, 0.027, 0.003,
                     0.069, 0.007, 0.019, 0.029, 0.009, 0.007, 0.0595, 0.042,
                     0.027, 0.034, 0.015, 0.034, 0.042, 0.248, 0.248, 0.363,
                     0.022, 0.018, 0.08, 0.033, 0.186, 0.034, 0.111, 0.039,
                     0.068, 0.071, 0.12, 0.13, 0.063, 0.012, 0.029, 0.043,
                     0.008, 0.06, 0.014, 0.029, 0.027, 0.047, 0.064, 0.036,
                     0.081, 0.254, 0.127, 0.037, 0.087, 0.0064, 0.154, 0.029,
                     0.191, 0.089, 0.014, 0.072, 0.262, 0.039, 0.084, 0.084,
                     0.041, 0.054, 0.042, 0.145, 0.118, 0.062, 0.094, 0.109,
                     0.08, 0.073, 0.103, 0.169, 0.08, 0.537, 0.026, 0.013,
                     0.065, 0.105, 0.007, 0.172, 0.232, 0.157, 0.1, 0.1234,
                     0.131, 0.099, 0.087, 0.193, 0.057, 0.052, 0.027, 0.2013,
                     0.043, 0.006, 0.07, 0.2676, 0.2127, 0.022, 0.036, 0.375,
                     0.107, 0.033, 0.033, 0.064, 0.093, 0.048, 0.119, 0.174,
                     0.253, 0.077, 0.144, 0.092, 0.427, 0.053, 0.092, 0.122,
                     0.064, 0.121, 0.135, 0.004, 0.354, 0.174, 0.038, 0.064,
                     0.062, 0.13, 0.124, 0.088, 0.4, 0.208, 0.184, 0.274,
                     0.393, 0.404, 0.042, 0.0199, 0.1049, 0.0018, 0.0271,
                     0.6163, -0.3697, 0.2915, 0.0339, 0.0582, 0.2344, 0.3669,
                     0.1076, 0.1414, 0.0197, 0.0405, 0.1106, 0.1688, 0.05,
                     0.0538, 0.1157, 0.0577, 0.0336, 0.0577, 0.0773, 0.0193,
                     0.0752, 0.0186, 0.0164, 0.0165, 0.0793, 0.0784, 0.0117,
                     0.0193, 0.0186, 0.0101, 0.0293, 0.0381, 0.0623, 0.089,
                     0.039, 0.0114, 0.0034, 0.0151, 0.0034, 0.2248, 0.0836,
                     0.1132, 0.3369, 0.3031, 0.3492, 0.0089, 0.2267, 0.057,
                     0.0773, 0.0909, 0.0588, 0.1511, 0.0504, 0.1526, 0.3919,
                     0.0957, 0.0288, 0.3161, 0.3054, 0.0568, 0.0098, 0.0285,
                     0.0448, 0.0326, 0.0245, 0.0514, 0.0123, 0.0178, 0.4843,
                     0.221, 0.3331, 0.3071, 0.5017, 0.3626, 0.3092, 0.0829,
                     0.0847, 0.1855, 0.029, 0.0202, 0.0083, 0.0245, 0.0086,
                     0.0086, 0.0202, 0.095, 0.069, 0.0022, 0.0275, 0.048,
                     0.009, 0.063, 0.127, 0.0286, 0.0355, 0.0286, 0.04, 0.081,
                     0.06, 0.04, 0.688, 0.046, 0.073, 0.109, 0.135, 0.102,
                     0.128, 0.183, 0.059, 0.068, 0.446, 0.093, 0.421, 0.0778,
                     0.038, 0.0185, 0.0108, 0.0249, 0.0497, 0.0456, 0.0177,
                     0.0395, 0.0018, 0.0484, 0.0295, 0.0046, 0.0013, 0.0514,
                     0.064, 0.0081, 0.061, 0.0212, 0.0472, 0.0087, 0.0917,
                     0.0288, 0.0626, 0.0069, 0.0006, 0.0485, 0.0259, 0.0272,
                     0.0006, 0.0154, 0.0043, 0.0851, 0.0723, 0.0355, 0.1804,
                     0.0551, 0.0732, 0.2119, 0.1285, 0.1215, 0.1004, 0.1857,
                     0.052, 0.052, 0.005, 0.039, 0.039, 0.089, 0.053, 0.0311,
                     0.038, 0.014, 0.064, 0.047, 0.02, 0.021, 0.059, 0.038,
                     0.0244, 0.02, 0.048, 0.048, 0.046, 0.149, 0.0174, 0.028,
                     0.0195, 0.018, 0.014, 0.0402, 0.0603, 0.0498, 0.0833,
                     0.0371, 0.0182, 0.0392, 0.0639, 0.0256, 0.016, 0.0396,
                     0.0384, 0.0231, 0.0131, 0.252, 0.237, 0.0366, 0.22, 0.098,
                     0.128, 0.204, 0.211, 0.0122, 0.0354, 0.0195, 0.0332,
                     0.016, 0.016, 0.02, 0.023, 0.023, 0.0146, 0.01054, 0.0238,
                     0.03214, 0.0154, 0.0289, 0.01953, 0.0193, 0.01923, 0.023,
                     0.0124, 0.0167, 0.0312, 0.01654, 0.03159, 0.05347,
                     0.18181, 0.19607, 0.06896
                     ])
    # X_br = abs(X_br)
    G_br = np.zeros(len(X_br))
    B_br = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0.054, 0, 0.139, 0, 1.127, 0.018, 0.07, 0.014, 
                     0.033, 0.081, 0.013, 0.018, 0.118, 0.016, 0.097, 0.121, 
                     0.035, 0.048, 1.28, 0.036, 0.151, 0.015, 0.098, 0.28, 
                     0.05, 0.018, 0.134, 0.034, 0.065, 0.014, 0, 0.013, 0.02, 
                     0.026, 0.042, 0.008, 0.002, 0.003, 0.043, 0.008, 0.007, 
                     0.02, 0.048, 0.033, 0.016, 0.02, 0.04, 0.404, 0.022, 0, 
                     0.02, 0.036, 0.806, 0.035, 0.032, 0.016, 0.039, 0.037, 
                     0.312, 0.411, 0.69, 0.073, 0.013, 0.007, 0.043, 0.049, 
                     0.036, 0.035, 0.047, 0.082, 0.036, 0.071, 0.005, 0.015, 
                     0.485, 0.203, 0.013, 0.026, 0.028, 0.023, 0.062, 0.028, 
                     0.068, 0.047, 0.011, 0, 0.03, 0.018, 0.05, 0.379, 0.004, 
                     0.007, 0.038, 0.029, 0.022, 0.011, 0, 0.051, 0.058, 0.53,
                     0.53, 0.128, 0.183, 0.092, 0.014, 0.024, 0.031, 0.039, 
                     0.017, 0.012, 0.043, 0.007, 0.012, 0.015, 0.007, 0.015, 
                     0.017, 0.002, 0.044, 0.022, 0.004, 0.027, 0.008, 0.044, 
                     0.015, 0.011, 0.047, 0.026, 0.021, 0.031, 0.036, 0.05, 
                     0.006, 0.004, 0.001, 0.017, 0, 0, 0, 0, 0, 0, 0.029, 
                     0.054, 0.117, 0.149, 0, 0.25, 0.185, 0.321, 0.33, 0.335,
                     0.171, 0.095, 0.208, 0.095, 0.126, 0.03, 0.122, 0.03, 
                     0.026, 0.026, 0.127, 0.125, 0.289, 0.03, 0.03, 0, 0.18, 
                     0.258, 0.385, 0.54, 0.953, 0.284, 0.021, 0.126, 0.021, 
                     0.081, 0.123, 0.684, 0.519, 0.463, 0.538, 0.119, 0.342, 
                     0.767, 0.119, 0.135, 0.087, 0.223, 0.074, 0.074, 0.076, 
                     0.141, 0.19, 0.04, 0.045, 0.388, 0.069, 0.19, 0.277, 
                     0.236, 1.662, 3.597, 0, 0.029, 0.063, 0.031, 0.049, 
                     0.043, 0.069, 0.048, 0.054, 0.047, 0.05, 0.029, 0.084, 
                     0, 0.115, 0.164, 0.115, 0.115, 0, 0, 0, 6.2, 0, 0, 0, 0, 
                     0, 0.5, 0.36, 0.5, 0.004, 0.01, 0.009, 0.006, 0, 0.025, 
                     0.008, 0.032, 0.009, 0.005, 0.014, 0, 0.008, 0, 0, 0, 0, 
                     0.086, 0, 0.02, 0.002, 0.026, 0.018, 0, 0.02, 0.832, 5.2, 
                     0.43, 0.503, 0.402, 1, 0.33, 0.48, 0.86, 0, 0, 0.186, 
                     1.28, 0, 0.81, 0, 1.364, 3.57, 0, 0.144, 0, 0.8, 0, 
                     0.009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, -0.087, 0, 0, 0, 0, 0, 0, 0, 0, -0.057, 
                     -0.033, 0, 0, 0, 0, 0, 0, 0, -0.012, 0, 0, -0.01, -0.364,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0
                     ])

    tr_ratio = np.array([1.0082, 0, 0.9668, 0.9796, 1.0435, 0.9391, 1.0435,
                         1.0435, 1.0435, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                         1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0.9583,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0.947, 0.956, 0.971, 0.948, 
                         0.959, 1.046, 0.985, 0.9561, 0.971, 0.952, 0.943, 
                         1.01, 1.008, 1, 0.975, 1.017, 1, 1, 1, 1, 1.015, 
                         0.967, 1.01, 1.05, 1, 1.0522, 1.0522, 1.05, 0.975, 
                         1, 1.035, 0.9565, 1, 1.05, 1.073, 1.05, 1.0506, 0.975,
                         0.98, 0.956, 1.05, 1.03, 1.03, 0.985, 1, 1.03, 1.01, 
                         1.05, 1.03, 1, 0.97, 1, 1.02, 1.07, 1.02, 1, 1.0223, 
                         0.9284, 1, 1, 1, 0.95, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                         0.942, 0.965, 0.95, 0.942, 0.942, 0.9565
                         ])
    from_bus_nr = np.array([37, 9001, 9001, 9001, 9005, 9005, 9005, 9005, 9005,
                            9006, 9006, 9006, 9012, 9012, 9002, 9021, 9021, 
                            9002, 9023, 9023, 9007, 9007, 9007, 9003, 9003, 
                            9003, 9003, 9044, 9004, 9004, 9004, 9003, 9003, 
                            9003, 9003, 9003, 9012, 9053, 1, 2, 2, 3, 3, 3, 4,
                            5, 7, 7, 8, 8, 9, 11, 12, 13, 14, 15, 15, 15, 16, 
                            19, 19, 20, 20, 21, 22, 23, 24, 25, 26, 26, 33, 33,
                            33, 33, 34, 35, 35, 35, 36, 37, 37, 37, 37, 37, 37,
                            38, 38, 39, 40, 41, 41, 41, 42, 43, 43, 43, 44, 44,
                            45, 45, 46, 47, 47, 48, 49, 51, 52, 53, 54, 55, 57,
                            57, 58, 59, 60, 62, 62, 63, 69, 69, 70, 70, 71, 71,
                            72, 72, 73, 73, 74, 74, 76, 77, 77, 77, 77, 78, 78,
                            79, 80, 81, 81, 85, 86, 86, 89, 90, 91, 91, 92, 92, 
                            94, 97, 97, 97, 98, 98, 99, 99, 99, 99, 100, 102, 
                            103, 104, 104, 105, 105, 108, 109, 109, 109, 110, 
                            112, 115, 116, 117, 118, 118, 1201, 118, 119, 119, 
                            122, 122, 123, 123, 125, 126, 126, 126, 126, 126, 
                            126, 127, 127, 127, 128, 128, 129, 129, 130, 130, 
                            130, 130, 133, 133, 133, 133, 134, 134, 135, 136, 
                            136, 137, 137, 137, 137, 139, 140, 140, 140, 140, 
                            140, 140, 141, 142, 143, 143, 145, 145, 146, 148, 
                            148, 152, 153, 154, 154, 155, 157, 158, 158, 162, 
                            162, 163, 165, 167, 172, 172, 173, 173, 173, 175, 
                            175, 176, 177, 178, 178, 181, 181, 184, 186, 187, 
                            188, 189, 189, 190, 190, 191, 192, 193, 193, 194, 
                            194, 195, 196, 196, 197, 197, 198, 198, 198, 198, 
                            199, 199, 200, 201, 203, 204, 205, 206, 206, 212, 
                            213, 214, 214, 215, 216, 217, 217, 217, 219, 220, 
                            220, 220, 221, 222, 224, 224, 225, 226, 227, 228, 
                            228, 228, 229, 231, 231, 232, 234, 234, 235, 241, 
                            240, 242, 242, 243, 243, 244, 245, 245, 246, 247, 
                            248, 249, 3, 3, 3, 7, 7, 10, 12, 15, 16, 21, 24, 
                            36, 45, 45, 62, 63, 73, 81, 85, 86, 87, 114, 116,
                            121, 122, 130, 130, 132, 141, 142, 143, 143, 145,
                            151, 153, 155, 159, 160, 163, 164, 182, 189, 193,
                            195, 200, 201, 202, 204, 209, 211, 218, 223, 229,
                            234, 238, 196, 119, 120, 7002, 7003, 7061, 7062, 
                            7166, 7024, 7001, 7130, 7011, 7023, 7049, 7139, 
                            7012, 7017, 7039, 7057, 7044, 7055, 7071
                            ])
    to_bus_nr = np.array([9001, 9005, 9006, 9012, 9051, 9052, 9053, 9054, 9055,
                          9007, 9003, 9003, 9002, 9002, 9021, 9023, 9022, 9024,
                          9025, 9026, 9071, 9072, 9003, 9031, 9032, 9033, 9044,
                          9004, 9041, 9042, 9043, 9034, 9035, 9036, 9037, 9038,
                          9121, 9533, 5, 6, 8, 7, 19, 150, 16, 9, 12, 131, 11, 
                          14, 11, 13, 21, 20, 15, 37, 89, 90, 42, 21, 87, 22, 
                          27, 24, 23, 25, 319, 26, 27, 320, 34, 38, 40, 41, 42,
                          72, 76, 77, 88, 38, 40, 41, 49, 89, 90, 41, 43, 42, 
                          48, 42, 49, 51, 46, 44, 48, 53, 47, 54, 60, 74, 81, 
                          73, 113, 107, 51, 52, 55, 54, 55, 57, 58, 63, 59, 61, 
                          62, 64, 144, 526, 211, 79, 71, 528, 72, 73, 77, 531, 
                          76, 79, 88, 562, 77, 78, 80, 552, 609, 79, 84, 211, 
                          211, 194, 195, 86, 87, 323, 91, 92, 94, 97, 103, 105,
                          97, 100, 102, 103, 100, 102, 107, 108, 109, 110, 102, 
                          104, 105, 108, 322, 107, 110, 324, 110, 113, 114, 
                          112, 114, 122, 120, 118, 119, 1201, 120, 121, 120, 
                          121, 123, 125, 124, 125, 126, 127, 129, 132, 157, 
                          158, 169, 128, 134, 168, 130, 133, 130, 133, 132, 
                          151, 167, 168, 137, 168, 169, 171, 135, 184, 136, 
                          137, 152, 140, 181, 186, 188, 172, 141, 142, 145, 
                          146, 147, 182, 146, 143, 145, 149, 146, 149, 147, 
                          178, 179, 153, 161, 156, 183, 161, 159, 159, 160, 
                          164, 165, 164, 166, 169, 173, 174, 174, 175, 176, 
                          176, 179, 177, 178, 179, 180, 138, 187, 185, 188, 
                          188, 138, 208, 209, 231, 240, 192, 225, 205, 208, 
                          219, 664, 219, 197, 210, 198, 211, 202, 203, 210, 
                          211, 200, 210, 210, 204, 211, 205, 206, 207, 208, 
                          215, 214, 215, 242, 216, 217, 218, 219, 220, 237, 
                          218, 221, 238, 223, 237, 225, 226, 191, 231, 231, 
                          229, 231, 234, 190, 232, 237, 233, 235, 237, 238, 
                          237, 281, 245, 247, 244, 245, 246, 246, 247, 247, 
                          248, 249, 250, 1, 2, 4, 5, 6, 11, 10, 17, 15, 20, 23,
                          35, 44, 46, 61, 64, 74, 88, 99, 102, 94, 207, 124, 
                          115, 157, 131, 150, 170, 174, 175, 144, 148, 180,
                          170, 183, 156, 117, 124, 137, 155, 139, 210, 196, 
                          212, 248, 69, 211, 2040, 198, 212, 219, 224, 230, 
                          236, 239, 2040, 1190, 1200, 2, 3, 61, 62, 166, 24, 1,
                          130, 11, 23, 49, 139, 12, 17, 39, 57, 44, 55, 71
                          ])


    br_f = np.array([np.flatnonzero(bus_nrs == bus)[0] for bus in from_bus_nr])
    br_t = np.array([np.flatnonzero(bus_nrs == bus)[0] for bus in to_bus_nr])

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    branch['Tap'] = tr_ratio

    return bus, branch, gen


def polish_sp(rm_line=None):

    n_bus = 2736

    Sbase = 100

    bus_nrs = np.arange(1,2737)


    bus_type = np.ones(n_bus)
    pv = np.array([  16,   25,   26,   28,   41,   42,   43,   53,   54,   55,   56,
             57,   76,   77,   80,   96,  102,  103,  104,  105,  114,  115,
            116,  123,  124,  125,  130,  131,  132,  133,  144,  145,  146,
            148,  149,  153,  154,  156,  162,  163,  178,  203,  204,  213,
            219,  220,  238,  245,  308,  309,  310,  312,  313,  321,  364,
            365,  366,  367,  384,  385,  442,  449,  451,  484,  535,  536,
            550,  558,  559,  572,  625,  663,  664,  684,  690,  715,  739,
            762,  791,  795,  806,  807,  844,  876,  899,  974,  975,  979,
           1000, 1001, 1004, 1005, 1006, 1012, 1014, 1023, 1024, 1025, 1112,
           1113, 1114, 1151, 1152, 1153, 1165, 1178, 1179, 1187, 1245, 1246,
           1247, 1285, 1286, 1287, 1341, 1342, 1350, 1351, 1352, 1362, 1373,
           1374, 1375, 1413, 1414, 1431, 1432, 1433, 1434, 1438, 1460, 1643,
           1671, 1672, 1673, 1683, 1684, 1775, 1792, 1793, 1835, 1837, 1844,
           1871, 1872, 1905, 1906, 1925, 1927, 1928, 1943, 1946, 1956, 1961,
           1963, 1964, 1990, 1996, 2000, 2001, 2006, 2013, 2014, 2034, 2038,
           2040, 2045, 2055, 2062, 2069, 2089, 2090, 2092, 2095, 2117, 2139,
           2148, 2174, 2201, 2202, 2207, 2213, 2214, 2227, 2234, 2252, 2253,
           2264, 2276, 2292, 2294, 2295, 2327, 2330, 2342, 2353, 2354, 2359,
           2365, 2369, 2430, 2437, 2447, 2450, 2452, 2458, 2466, 2469, 2470,
           2479, 2480, 2481, 2482, 2484, 2487, 2490, 2496, 2499, 2508, 2510,
           2522, 2525, 2528, 2535, 2538, 2541, 2543, 2545, 2546, 2553, 2555,
           2556, 2562, 2563, 2567, 2570, 2572, 2573, 2577, 2581, 2588, 2605,
           2624, 2627, 2631, 2633, 2642, 2647, 2648, 2650, 2652, 2653, 2654,
           2657, 2661, 2666, 2667, 2673, 2674, 2680, 2683, 2689, 2690, 2694,
           2700, 2701, 2711, 2713, 2715, 2718, 2719, 2726, 2727, 2729])
    bus_type[pv] = 2
    bus_type[27] = 3


    bus_nr = np.arange(n_bus)

    Vm = np.array([1.1010186, 1.1060162, 1.0848541, 1.0923320, 1.0923630,
                    1.0653306, 1.0654455, 1.0256415, 1.0256531, 1.0586771,
                    1.0585950, 1.0585464, 1.0585095, 1.0183879, 1.0315259,
                    1.0315259, 1.0932112, 1.0932634, 1.0968490, 1.0968889,
                    1.0778181, 1.0778181, 1.0563162, 1.0981385, 1.0325941,
                    1.1100000, 1.1100000, 1.0461868, 1.0461977, 1.0661645,
                    1.0438875, 1.0557975, 1.0992548, 1.0766806, 1.0798203,
                    1.0876123, 1.0876009, 1.0668339, 1.1100000, 1.1098870,
                    1.0237808, 1.0911339, 1.0910418, 1.0274551, 1.0274089,
                    1.1065267, 1.0378593, 1.0378290, 1.0341541, 1.0744468,
                    1.0744293, 1.0381233, 1.0381438, 1.0924320, 1.0923941,
                    1.0448325, 1.0448051, 1.0849742, 1.0849851, 1.1018100,
                    1.0925329, 1.0924964, 1.0484928, 1.0484430, 1.0810500,
                    1.0825042, 1.0750296, 1.1010761, 1.0960299, 1.0958743,
                    1.0988119, 1.0988429, 1.0958718, 1.0958515, 1.0981431,
                    1.0980313, 1.1017006, 1.1001979, 1.0917918, 1.0918096,
                    1.0324377, 1.0881440, 1.0979531, 1.0979533, 1.1019970,
                    1.1019728, 1.0921611, 1.0921486, 1.1032657, 1.1032630,
                    1.0475653, 1.0475601, 1.0935069, 1.0935303, 1.0927523,
                    1.0927385, 1.0988887, 1.0988852, 1.0960777, 1.0960847,
                    1.0914730, 1.0914416, 1.0995351, 1.0994826, 1.0951885,
                    1.0951370, 1.0993639, 1.0993779, 1.0935543, 1.0870297,
                    1.0870594, 1.0929400, 1.0929963, 1.0414756, 1.0996551,
                    1.0996683, 1.0980564, 1.0980782, 1.0496156, 1.0496497,
                    1.0499794, 1.0500000, 1.0928602, 1.0993585, 1.0992582,
                    1.0363847, 1.0363725, 1.1013954, 1.1013663, 1.1018746,
                    1.1008625, 1.0980639, 1.1100000, 1.1099315, 1.0810731,
                    1.0828301, 1.0065862, 1.0066041, 1.0954147, 1.1000855,
                    1.1000855, 1.0997348, 1.0776819, 1.0776819, 1.1088684,
                    1.1088509, 1.1100000, 1.1099376, 1.0401684, 1.0401565,
                    1.0697070, 1.0969163, 1.0968690, 1.0864589, 1.0864932,
                    1.0043516, 1.0043458, 1.0991142, 1.0991142, 1.0126700,
                    1.0153659, 1.0153480, 1.1100000, 1.1100000, 1.1100000,
                    1.0904059, 1.0992392, 1.0934021, 1.0933821, 1.0280759,
                    1.0280759, 1.0678718, 1.0678575, 1.0911492, 1.0802612,
                    1.0802658, 1.0817101, 1.0817101, 1.1099162, 1.0673522,
                    1.0732574, 1.0910728, 1.0378187, 1.0378411, 1.0320725,
                    1.0320886, 1.0799827, 1.0800155, 1.0758780, 1.0758447,
                    1.0182950, 1.0770296, 1.0770074, 1.0326582, 1.0326384,
                    1.0748167, 1.0748330, 1.0292850, 1.0293306, 1.0688896,
                    1.0688729, 1.0716109, 1.0715607, 1.0342258, 1.0342303,
                    1.0867692, 1.0867775, 1.0888166, 1.0763513, 1.0944055,
                    1.0655484, 1.0903140, 1.0968887, 1.1118352, 1.1129922,
                    1.1125137, 1.1114167, 1.0812449, 1.0810050, 1.1012585,
                    1.0943315, 1.0912687, 1.0754351, 1.1051233, 1.0928473,
                    1.1035287, 1.0926342, 1.0926720, 1.1120912, 1.0943126,
                    1.0838177, 1.0970947, 1.0734703, 1.0796816, 1.0868262,
                    1.1086811, 1.0862746, 1.0935310, 1.0840654, 1.0860973,
                    1.1000107, 1.1139775, 1.1140714, 1.1061558, 1.0983181,
                    1.0790447, 1.0886183, 1.0888205, 1.0955405, 1.0656001,
                    1.1080612, 1.0885084, 1.0850344, 1.0778039, 1.0952060,
                    1.0954886, 1.0900687, 1.0900172, 1.1039829, 1.0647876,
                    1.0892538, 1.0879100, 1.0846324, 1.0967395, 1.0890012,
                    1.1007233, 1.0833557, 1.0851905, 1.1114105, 1.1022215,
                    1.0663813, 1.0833853, 1.0792157, 1.0900879, 1.0889678,
                    1.1150578, 1.1084132, 1.0968772, 1.1069276, 1.0992968,
                    1.0926239, 1.0927560, 1.0788213, 1.0785542, 1.1044943,
                    1.0986013, 1.1155570, 1.1097537, 1.0905924, 1.1050274,
                    1.0845059, 1.0978273, 1.0934149, 1.0900309, 1.1173899,
                    1.0896499, 1.0821362, 1.0964546, 1.1058044, 1.1105660,
                    1.0936642, 1.1090796, 1.1086778, 1.0697496, 1.1174004,
                    1.1003819, 1.1148589, 1.1108519, 1.1093260, 1.1118607,
                    1.0971805, 1.1106712, 1.1155075, 1.1064786, 1.0973280,
                    1.1049384, 1.0811616, 1.0818273, 1.0817721, 1.0892829,
                    1.1153908, 1.1139769, 1.0975569, 1.1065538, 1.1088020,
                    1.1091726, 1.0989807, 1.0821851, 1.1091807, 1.1064358,
                    1.1060470, 1.1127952, 1.0912164, 1.1026807, 1.0814444,
                    1.0823415, 1.1020343, 1.0894284, 1.0887891, 1.0758718,
                    1.1070738, 1.0962765, 1.1018629, 1.0891147, 1.0892454,
                    1.0964870, 1.0725538, 1.0927210, 1.0827192, 1.1170831,
                    1.0825148, 1.0937702, 1.0999503, 1.0888460, 1.0873727,
                    1.0875699, 1.0873622, 1.0884634, 1.0785867, 1.0921222,
                    1.1006300, 1.1001956, 1.0716895, 1.1049361, 1.1119447,
                    1.0922113, 1.1187323, 1.1111137, 1.1200000, 1.0836579,
                    1.0882143, 1.0772135, 1.0953978, 1.0821949, 1.1091155,
                    1.1026527, 1.0769854, 1.0833934, 1.0739702, 1.0740079,
                    1.0827573, 1.1048396, 1.1058697, 1.0908721, 1.1073564,
                    1.1024838, 1.0979925, 1.0930522, 1.0944974, 1.0887198,
                    1.1182873, 1.1123844, 1.0982826, 1.0985644, 1.1081480,
                    1.0964649, 1.1093176, 1.0934458, 1.1113699, 1.1124238,
                    1.0896574, 1.0855065, 1.0712893, 1.0922543, 1.0877271,
                    1.1017689, 1.0797234, 1.0829924, 1.1012096, 1.0924407,
                    1.0876239, 1.0899952, 1.1200000, 1.0925659, 1.0816290,
                    1.0821314, 1.0929266, 1.0895389, 1.1097214, 1.1033462,
                    1.0779171, 1.0778691, 1.0833235, 1.0834043, 1.0896772,
                    1.0872979, 1.0829920, 1.0842798, 1.1163184, 1.1163234,
                    1.0811990, 1.0914726, 1.0888245, 1.1009728, 1.1123586,
                    1.0754308, 1.0839560, 1.0801964, 1.0937512, 1.0663961,
                    1.0978117, 1.0977405, 1.0808287, 1.0721308, 1.1105941,
                    1.1090101, 1.1095080, 1.0868686, 1.1051545, 1.0907525,
                    1.0916318, 1.0928766, 1.0646620, 1.0846958, 1.0865892,
                    1.1143200, 1.1036456, 1.0843581, 1.0997236, 1.0820515,
                    1.0958630, 1.0851716, 1.0943288, 1.0953998, 1.0954379,
                    1.0840634, 1.0978323, 1.1081021, 1.1092629, 1.0944641,
                    1.1056390, 1.1010698, 1.0773032, 1.0678034, 1.0888084,
                    1.0961327, 1.0861930, 1.1043653, 1.0712316, 1.0824331,
                    1.0950557, 1.0894320, 1.0841134, 1.0717061, 1.0731501,
                    1.0834579, 1.0714355, 1.0826447, 1.1159998, 1.1089331,
                    1.0885432, 1.0882752, 1.1116013, 1.0807277, 1.0808231,
                    1.0820240, 1.0790913, 1.0840391, 1.0846626, 1.0825229,
                    1.0804946, 1.0745799, 1.0815684, 1.0787391, 1.0755367,
                    1.0632968, 1.0897515, 1.0941115, 1.1048925, 1.1048598,
                    1.1065766, 1.1147437, 1.0833247, 1.1022804, 1.0860503,
                    1.0860952, 1.0777817, 1.0897715, 1.0941641, 1.0856459,
                    1.0870307, 1.0773042, 1.0994272, 1.0827819, 1.0828419,
                    1.0836187, 1.0818334, 1.0795905, 1.0762504, 1.0827367,
                    1.0778236, 1.0816953, 1.0795281, 1.0789356, 1.1009519,
                    1.0850146, 1.0830050, 1.0822870, 1.0842980, 1.0817632,
                    1.0821650, 1.0822680, 1.0826217, 1.0835483, 1.0832006,
                    1.0816260, 1.0877573, 1.0892241, 1.0820400, 1.0866030,
                    1.0775252, 1.0784713, 1.0888814, 1.0815603, 1.0986329,
                    1.0882670, 1.0882696, 1.0657851, 1.0822821, 1.0810281,
                    1.0985589, 1.0828560, 1.0822573, 1.1006760, 1.0859368,
                    1.1107038, 1.0927805, 1.0774990, 1.0858447, 1.1014154,
                    1.0967959, 1.0974439, 1.1045147, 1.1121571, 1.0943796,
                    1.1013174, 1.1161161, 1.0967189, 1.0873910, 1.0903515,
                    1.0872088, 1.0934413, 1.1116668, 1.0818192, 1.0812355,
                    1.0793210, 1.0983678, 1.0984547, 1.0786202, 1.0892973,
                    1.0535843, 1.0897643, 1.0711530, 1.0710879, 1.0850909,
                    1.0881552, 1.0534183, 1.0825496, 1.0817988, 1.0706283,
                    1.1026660, 1.1013164, 1.0617541, 1.0773867, 1.0840564,
                    1.0731761, 1.0736031, 1.0903330, 1.0523098, 1.0893993,
                    1.0888630, 1.0909008, 1.0839961, 1.0809465, 1.0908868,
                    1.0769894, 1.0916459, 1.0916054, 1.0758924, 1.0906854,
                    1.0790017, 1.0897788, 1.0740161, 1.0647216, 1.0725724,
                    1.0941580, 1.0780318, 1.0824934, 1.0782498, 1.0912689,
                    1.0788939, 1.0803569, 1.0488039, 1.0936539, 1.0937500,
                    1.0784985, 1.0666752, 1.1041513, 1.0521750, 1.0743080,
                    1.0889713, 1.0651714, 1.0789343, 1.0787686, 1.0827093,
                    1.1062253, 1.0698567, 1.0967902, 1.0830169, 1.0769417,
                    1.0776664, 1.0797642, 1.0884174, 1.0778131, 1.0769716,
                    1.0770893, 1.0767677, 1.0496165, 1.0489435, 1.0732144,
                    1.0824068, 1.0782908, 1.0783016, 1.1007585, 1.1003072,
                    1.0763565, 1.0784033, 1.0503279, 1.0527804, 1.0502804,
                    1.0505225, 1.0515987, 1.0682740, 1.0684260, 1.0896696,
                    1.0685109, 1.0725870, 1.0760108, 1.0943485, 1.0650914,
                    1.0668265, 1.0891400, 1.0864699, 1.0968180, 1.1007953,
                    1.1007374, 1.0873238, 1.0632703, 1.0869602, 1.0940559,
                    1.0939195, 1.0935497, 1.0925768, 1.0804221, 1.1029893,
                    1.0934139, 1.0934329, 1.0982344, 1.0924968, 1.0961950,
                    1.0936889, 1.0945913, 1.0959714, 1.0970639, 1.0939438,
                    1.0812058, 1.0818185, 1.0942235, 1.0792834, 1.0941163,
                    1.0572655, 1.0871718, 1.0888235, 1.0951011, 1.0951283,
                    1.0745186, 1.0718069, 1.0798094, 1.0635204, 1.0800658,
                    1.0798903, 1.0490971, 1.0828127, 1.0855382, 1.0726886,
                    1.0849572, 1.0825499, 1.0714496, 1.0829792, 1.0837796,
                    1.0839521, 1.0824910, 1.0149549, 1.0151430, 1.0802374,
                    1.0838441, 1.0792731, 1.0983618, 1.0693535, 1.0856708,
                    1.0857149, 1.0731576, 1.0881508, 1.0822647, 1.0774029,
                    1.0775292, 1.0856141, 1.0859948, 1.0943232, 1.0549316,
                    1.0602903, 1.0749082, 1.0818610, 1.0545023, 1.0572697,
                    1.0551566, 1.0731718, 1.0814618, 1.0815069, 1.0882824,
                    1.0739829, 1.0740789, 1.0824360, 1.0893353, 1.0755497,
                    1.0744393, 1.0571246, 1.0779930, 1.0779628, 1.0751753,
                    1.0749549, 1.0750796, 1.0909702, 1.0832133, 1.0736927,
                    1.0733052, 1.0716603, 1.0810650, 1.0810357, 1.0738908,
                    1.0820114, 1.0712731, 1.0830892, 1.0940977, 1.0745076,
                    1.0929237, 1.0927761, 1.0711179, 1.0929293, 1.0808349,
                    1.0593702, 1.0597400, 1.0603115, 1.0804174, 1.0828682,
                    1.0695421, 1.0696198, 1.0688854, 1.0696292, 1.0726827,
                    1.0816001, 1.0731987, 1.0857411, 1.0599007, 1.0755943,
                    1.0728190, 1.0834189, 1.0898163, 1.0590846, 1.0716373,
                    1.0816135, 1.0828650, 1.0903613, 1.0809984, 1.0804195,
                    1.0774291, 1.0805074, 1.0785352, 1.0950494, 1.0774053,
                    1.0653912, 1.0907349, 1.1022332, 1.1106405, 1.0800959,
                    1.0965498, 1.0748577, 1.0845146, 1.0965230, 1.0700368,
                    1.0729476, 1.0971845, 1.0608460, 1.0937528, 1.0859119,
                    1.0957470, 1.0966155, 1.0700460, 1.0779453, 1.0957747,
                    1.0911985, 1.0706592, 1.0712301, 1.0703805, 1.0709617,
                    1.0767869, 1.0768100, 1.1056474, 1.0827551, 1.0827571,
                    1.0793596, 1.0793926, 1.0794257, 1.0959128, 1.0958832,
                    1.0955471, 1.0955457, 1.0899372, 1.0900364, 1.0917781,
                    1.0917791, 1.0701918, 1.0788628, 1.0787890, 1.0873618,
                    1.0843737, 1.0843736, 1.0818057, 1.0817700, 1.0799302,
                    1.0799615, 1.0643340, 1.0740212, 1.0643481, 1.0849135,
                    1.0848423, 1.0901923, 1.0865275, 1.0659455, 1.0762017,
                    1.0771513, 1.0682389, 1.0682312, 1.0990898, 1.0991174,
                    1.0824721, 1.0819442, 1.0825121, 1.0819652, 1.1015237,
                    1.1015092, 1.0767091, 1.0771577, 1.0815776, 1.0816057,
                    1.0861690, 1.0788624, 1.0870708, 1.0856554, 1.0829685,
                    1.0896019, 1.0950704, 1.0950316, 1.0679550, 1.0679791,
                    1.0487023, 1.0487023, 1.0938089, 1.0862299, 1.0989595,
                    1.0991783, 1.0965612, 1.0920096, 1.0919258, 1.0964189,
                    1.0804356, 1.0803940, 1.0917513, 1.0917410, 1.0830622,
                    1.0853326, 1.0857831, 1.0862074, 1.0858202, 1.0862064,
                    1.0866022, 1.0866077, 1.0828883, 1.0915530, 1.0610632,
                    1.0611153, 1.0814795, 1.0814628, 1.0744291, 1.0744912,
                    1.0857599, 1.0857531, 1.0885452, 1.0885388, 1.0886139,
                    1.0885829, 1.0981519, 1.0973939, 1.0985243, 1.0965611,
                    1.0921013, 1.0812433, 1.0812765, 1.0934181, 1.0942049,
                    1.0765984, 1.0767454, 1.0833865, 1.0834362, 1.0768456,
                    1.0769140, 1.0957591, 1.0959010, 1.0635957, 1.0635538,
                    1.0775806, 1.0775552, 1.0857793, 1.0857635, 1.0683856,
                    1.0684284, 1.0985046, 1.0985027, 1.0957583, 1.0957975,
                    1.0872462, 1.0872154, 1.0894150, 1.0894222, 1.0881579,
                    1.0977934, 1.1035023, 1.1034279, 1.1051877, 1.1053226,
                    1.0897310, 1.0897426, 1.0750700, 1.0750704, 1.0987225,
                    1.0987493, 1.0836994, 1.0836948, 1.0818634, 1.0818999,
                    1.0945562, 1.0946905, 1.0809635, 1.0809216, 1.0847390,
                    1.0928435, 1.0819826, 1.0761829, 1.0850597, 1.0850857,
                    1.1022980, 1.1005301, 1.0843326, 1.0843168, 1.1028050,
                    1.1027837, 1.0942396, 1.0942408, 1.0741288, 1.0740451,
                    1.0691710, 1.0692317, 1.0980995, 1.0893353, 1.0936423,
                    1.1023904, 1.1023946, 1.0812473, 1.0812865, 1.0512239,
                    1.0512534, 1.0863303, 1.0863339, 1.0765009, 1.0765369,
                    1.0764979, 1.0852664, 1.0840441, 1.0840439, 1.0942399,
                    1.0942410, 1.0826059, 1.0826056, 1.0938606, 1.1018972,
                    1.1017714, 1.1096904, 1.1096857, 1.0858134, 1.0858306,
                    1.0919079, 1.0919104, 1.1096904, 1.1096860, 1.0822216,
                    1.0915599, 1.0756156, 1.0757186, 1.1024194, 1.1024928,
                    1.0914496, 1.0913996, 1.0917674, 1.0923010, 1.0871151,
                    1.0871883, 1.0791822, 1.0791804, 1.0858143, 1.0862208,
                    1.0845276, 1.0845097, 1.0583210, 1.0502012, 1.0502178,
                    1.0849120, 1.0849980, 1.0849444, 1.0889829, 1.0890065,
                    1.0873249, 1.0873801, 1.0781610, 1.0781509, 1.0954989,
                    1.0955113, 1.0684258, 1.0684391, 1.0817236, 1.0817234,
                    1.0843304, 1.0844007, 1.0843460, 1.0844001, 1.0930255,
                    1.0930393, 1.0720210, 1.0720538, 1.0945156, 1.0945988,
                    1.0923030, 1.0918182, 1.0731710, 1.0733851, 1.0612793,
                    1.0612588, 1.1019551, 1.1060698, 1.1020717, 1.1060688,
                    1.0955740, 1.0955865, 1.0978057, 1.0847016, 1.0978022,
                    1.0849497, 1.0889681, 1.0889573, 1.0901500, 1.0901256,
                    1.0763991, 1.0520826, 1.0948005, 1.0862757, 1.0977473,
                    1.0978891, 1.0978675, 1.0597211, 1.0598991, 1.0608271,
                    1.0609893, 1.0955890, 1.0955718, 1.0955888, 1.0955979,
                    1.0952762, 1.0957688, 1.0953099, 1.0957688, 1.0917587,
                    1.0950041, 1.0950784, 1.0950955, 1.0833324, 1.0833107,
                    1.0880851, 1.0976528, 1.0921876, 1.0905621, 1.0921875,
                    1.0981481, 1.0913627, 1.0803395, 1.0803247, 1.0803505,
                    1.0837543, 1.0845667, 1.0870340, 1.0870072, 1.0928729,
                    1.0929039, 1.0776482, 1.0777506, 1.0806526, 1.0775424,
                    1.0775460, 1.0768279, 1.0807879, 1.0806311, 1.0777791,
                    1.0805904, 1.0807599, 1.0775747, 1.0772535, 1.0772710,
                    1.0815746, 1.0806451, 1.0815723, 1.0806587, 1.1052172,
                    1.1054381, 1.0820300, 1.0820300, 1.0915826, 1.0916094,
                    1.0819854, 1.1029405, 1.0905176, 1.0896108, 1.0883058,
                    1.0680551, 1.0680336, 1.0894546, 1.0894546, 1.0966839,
                    1.0965522, 1.0249955, 1.0858186, 1.0844283, 1.0934605,
                    1.0844283, 1.0910421, 1.0910907, 1.0844278, 1.0844659,
                    1.0825485, 1.0895090, 1.0894498, 1.0895385, 1.0894574,
                    1.1015394, 1.1015612, 1.0942400, 1.0900422, 1.0900428,
                    1.0838353, 1.0838329, 1.0972532, 1.0972911, 1.1003508,
                    1.1003078, 1.0979915, 1.0979930, 1.0644485, 1.0644434,
                    1.0862902, 1.0862946, 1.0864163, 1.0894836, 1.0882162,
                    1.0881952, 1.0921548, 1.0923381, 1.0836502, 1.0836044,
                    1.0722129, 1.0726559, 1.0726632, 1.0728790, 1.0891423,
                    1.0891955, 1.0938594, 1.0862577, 1.0941891, 1.0941804,
                    1.0860776, 1.0861066, 1.0827632, 1.0827632, 1.0876629,
                    1.0876249, 1.0959115, 1.0959071, 1.0555892, 1.0555697,
                    1.0889161, 1.0793722, 1.0801915, 1.0801988, 1.0860748,
                    1.0939598, 1.0861062, 1.0940031, 1.0725077, 1.0707949,
                    1.0930815, 1.0930871, 1.1098337, 1.1098404, 1.0863600,
                    1.0864100, 1.0976700, 1.0977568, 1.0872655, 1.0872945,
                    1.0833550, 1.0834473, 1.0839232, 1.0978023, 1.0849250,
                    1.0847800, 1.0930606, 1.0934698, 1.0754852, 1.0889349,
                    1.0921200, 1.0923060, 1.0868497, 1.0868530, 1.0955076,
                    1.0955371, 1.0802594, 1.0802685, 1.0926222, 1.0852623,
                    1.0806770, 1.1175518, 1.0921304, 1.0875090, 1.0874852,
                    1.0833552, 1.0832928, 1.0899205, 1.0899291, 1.0921736,
                    1.0921958, 1.0871585, 1.0871677, 1.0814596, 1.0814441,
                    1.0826679, 1.0826672, 1.0799356, 1.0799168, 1.0657014,
                    1.0657117, 1.0877267, 1.0877131, 1.0815465, 1.0815438,
                    1.0791797, 1.0791726, 1.0919787, 1.0919033, 1.0859169,
                    1.0858977, 1.0900616, 1.0900728, 1.0895097, 1.0823288,
                    1.0895056, 1.0823248, 1.0969236, 1.0969003, 1.0810577,
                    1.0895259, 1.0810721, 1.0895259, 1.0762562, 1.0762622,
                    1.0601475, 1.0601689, 1.0837654, 1.0837554, 1.0825540,
                    1.0825520, 1.0610472, 1.0610471, 1.0919865, 1.0827785,
                    1.0915533, 1.0857386, 1.0857221, 1.0778591, 1.0853569,
                    1.0778959, 1.0816999, 1.0819931, 1.0623641, 1.0623076,
                    1.0895449, 1.0910792, 1.0956417, 1.0747060, 1.0746880,
                    1.1155039, 1.0902783, 1.1157374, 1.0908411, 1.0533975,
                    1.0533993, 1.0919812, 1.0991515, 1.0801196, 1.0801108,
                    1.0918046, 1.0918990, 1.0900550, 1.0900508, 1.1003732,
                    1.0994727, 1.1003687, 1.1000730, 1.0826835, 1.0825766,
                    1.0766098, 1.1051037, 1.1050193, 1.0842323, 1.0841805,
                    1.0882911, 1.0883240, 1.0948916, 1.0949131, 1.0775805,
                    1.0836704, 1.0918230, 1.0918365, 1.0989479, 1.0988711,
                    1.0728748, 1.0728548, 1.0777966, 1.0807983, 1.0948312,
                    1.0949197, 1.0857346, 1.0857372, 1.0900249, 1.0861771,
                    1.0900507, 1.0867866, 1.0867103, 1.0816724, 1.0809890,
                    1.0817725, 1.0810280, 1.0900215, 1.0899707, 1.0774596,
                    1.0776190, 1.0837769, 1.0846190, 1.0826167, 1.0826793,
                    1.0918566, 1.0918514, 1.0829836, 1.0829618, 1.0944697,
                    1.0944439, 1.0859088, 1.0859327, 1.0850509, 1.0850507,
                    1.0761731, 1.0761830, 1.0566584, 1.0566523, 1.0862927,
                    1.0862933, 1.0873686, 1.0854614, 1.0972679, 1.0988890,
                    1.0975138, 1.0794131, 1.0793936, 1.0856725, 1.0856451,
                    1.1065452, 1.1065430, 1.1077622, 1.1097065, 1.1079229,
                    1.1099813, 1.0658999, 1.0658907, 1.0920550, 1.0920611,
                    1.0804851, 1.0804514, 1.0848975, 1.0849381, 1.0680207,
                    1.0680258, 1.0888939, 1.0888213, 1.0722186, 1.0722202,
                    1.0813539, 1.0813460, 1.0661562, 1.0661918, 1.0858941,
                    1.0858947, 1.0919886, 1.0907135, 1.0920044, 1.0907345,
                    1.1002640, 1.1002463, 1.0707720, 1.0708201, 1.0881721,
                    1.0882660, 1.1039971, 1.1039187, 1.0917669, 1.0917538,
                    1.0859002, 1.0858319, 1.0826809, 1.0826638, 1.0926854,
                    1.0937335, 1.0744293, 1.0744338, 1.0976641, 1.0913781,
                    1.0976879, 1.0755171, 1.0755196, 1.0852291, 1.0852248,
                    1.0705818, 1.0706444, 1.0732095, 1.0732275, 1.0900507,
                    1.0864284, 1.0764898, 1.0765499, 1.0923298, 1.0940939,
                    1.0945600, 1.0954773, 1.0954979, 1.0933773, 1.0934518,
                    1.0914531, 1.1087687, 1.1041086, 1.1072008, 1.1041239,
                    1.1072008, 1.0886253, 1.0886277, 1.0751742, 1.0751742,
                    1.0873316, 1.0873853, 1.0693983, 1.0694583, 1.0957536,
                    1.0957886, 1.0661496, 1.0661471, 1.0877348, 1.0877850,
                    1.0879637, 1.0885782, 1.0879620, 1.0887509, 1.0781192,
                    1.0780898, 1.0717072, 1.0685806, 1.0717518, 1.0678977,
                    1.0762204, 1.0762066, 1.0778525, 1.0770727, 1.0825266,
                    1.0824709, 1.0862079, 1.0862553, 1.0818653, 1.0829320,
                    1.0819127, 1.0832180, 1.0920629, 1.0920517, 1.0934613,
                    1.0934795, 1.0812826, 1.0812452, 1.0850385, 1.0856325,
                    1.0718545, 1.0719324, 1.0798722, 1.0799792, 1.0800082,
                    1.0801248, 1.0848643, 1.0848039, 1.0901436, 1.0900852,
                    1.0775927, 1.0837084, 1.0912571, 1.0912885, 1.0770712,
                    1.0770586, 1.0465736, 1.0466027, 1.0861979, 1.0861256,
                    1.0721213, 1.0721444, 1.0993135, 1.0992812, 1.0900373,
                    1.0900863, 1.0974798, 1.0973826, 1.0899601, 1.0899254,
                    1.0649277, 1.0649601, 1.0714165, 1.0714681, 1.0918599,
                    1.0918889, 1.0826362, 1.0826550, 1.0825957, 1.0826037,
                    1.0725323, 1.0724623, 1.0751823, 1.0751782, 1.0874038,
                    1.0873758, 1.0665976, 1.0665940, 1.0917930, 1.0917995,
                    1.0842248, 1.0842745, 1.0813912, 1.0813415, 1.0969156,
                    1.0965909, 1.0829444, 1.0829702, 1.0987061, 1.0988294,
                    1.0936513, 1.0936470, 1.0938398, 1.0938175, 1.0860781,
                    1.0861972, 1.0835307, 1.0835724, 1.0846559, 1.0844961,
                    1.0875200, 1.0851882, 1.0851641, 1.0755648, 1.0755553,
                    1.0934433, 1.0934373, 1.0934383, 1.0934383, 1.0912486,
                    1.0922936, 1.0989802, 1.0989958, 1.0761758, 1.0761708,
                    1.0920920, 1.1175501, 1.0920918, 1.0970823, 1.0971191,
                    1.0979685, 1.0980003, 1.0771297, 1.0771620, 1.0918857,
                    1.0917573, 1.0774982, 1.0776255, 1.0937176, 1.0936817,
                    1.0690065, 1.0707858, 1.0707512, 1.0943437, 1.0944072,
                    1.0666805, 1.0666772, 1.0852778, 1.0852907, 1.0945396,
                    1.0945988, 1.1096068, 1.1004920, 1.0973965, 1.0948216,
                    1.0948452, 1.0965518, 1.0965395, 1.0898418, 1.0898844,
                    1.0895843, 1.1019676, 1.1018879, 1.0888958, 1.0884811,
                    1.0725217, 1.0725199, 1.0863213, 1.0863096, 1.0880413,
                    1.0887711, 1.0824507, 1.0859768, 1.0702267, 1.0700988,
                    1.0899872, 1.0899353, 1.0918135, 1.1035159, 1.0920304,
                    1.1040387, 1.0910588, 1.0918534, 1.0913828, 1.0789189,
                    1.0788841, 1.1003826, 1.1004482, 1.0899154, 1.0964587,
                    1.0903071, 1.0968147, 1.0889777, 1.0890286, 1.0802479,
                    1.0802122, 1.0848444, 1.0900995, 1.0835227, 1.0835401,
                    1.0900118, 1.0891810, 1.0900172, 1.0893195, 1.0715711,
                    1.0715104, 1.0880604, 1.0879869, 1.0894830, 1.0894495,
                    1.0843182, 1.0957789, 1.0844032, 1.0865371, 1.0865165,
                    1.0799110, 1.0799276, 1.0584295, 1.0584515, 1.0938124,
                    1.0934820, 1.0530244, 1.0531785, 1.0857402, 1.0857559,
                    1.0743219, 1.0742935, 1.0670038, 1.0670250, 1.0692950,
                    1.0692999, 1.0687454, 1.0687706, 1.0953501, 1.0952895,
                    1.0886097, 1.0884862, 1.0929442, 1.0928228, 1.0844834,
                    1.0845170, 1.0809407, 1.0808510, 1.0712297, 1.0706601,
                    1.0818317, 1.0818027, 1.0826362, 1.0826297, 1.0856205,
                    1.0856133, 1.0903216, 1.0903196, 1.0758115, 1.0758989,
                    1.0814384, 1.0815078, 1.0982061, 1.0983879, 1.0855561,
                    1.0855364, 1.0826989, 1.0829994, 1.0829378, 1.0832516,
                    1.0703741, 1.0704231, 1.0907003, 1.0907128, 1.0742110,
                    1.0739147, 1.1012540, 1.0992973, 1.1058562, 1.1057215,
                    1.1064335, 1.1063963, 1.0934002, 1.0861223, 1.0934277,
                    1.0827604, 1.0827431, 1.0994852, 1.1064988, 1.0692367,
                    1.0692773, 1.0815136, 1.0815136, 1.1026594, 1.1025017,
                    1.0622225, 1.0620722, 1.0828273, 1.0833415, 1.0811465,
                    1.0811465, 1.0808996, 1.0776840, 1.0779933, 1.0779518,
                    1.0825599, 1.0825802, 1.0823332, 1.0823354, 1.0821081,
                    1.0820505, 1.0936257, 1.0878792, 1.0880047, 1.0665163,
                    1.0665085, 1.0954808, 1.0955896, 1.0855353, 1.0855251,
                    1.1062750, 1.1009261, 1.1105307, 1.0919906, 1.0920332,
                    1.0602586, 1.0987239, 1.0704637, 1.0704450, 1.1082967,
                    1.1082498, 1.0903186, 1.0903706, 1.0880932, 1.0880670,
                    1.0933371, 1.0933254, 1.0828418, 1.0828562, 1.0974078,
                    1.1089652, 1.0942413, 1.0959290, 1.0932550, 1.0580837,
                    1.0580770, 1.0609507, 1.0609929, 1.0959507, 1.0958792,
                    1.0990452, 1.0990000, 1.0874350, 1.0919706, 1.0919598,
                    1.0554008, 1.0848239, 1.0824689, 1.0856274, 1.0924449,
                    1.0856321, 1.0924487, 1.0972883, 1.0972925, 1.0996046,
                    1.0995568, 1.0843941, 1.0844037, 1.0803653, 1.0803956,
                    1.0880037, 1.0880161, 1.0824716, 1.0860344, 1.0903183,
                    1.0902023, 1.0877756, 1.0877729, 1.0860445, 1.0860809,
                    1.0917300, 1.0917460, 1.0925188, 1.0925457, 1.0993466,
                    1.0992989, 1.0824329, 1.0824660, 1.0917523, 1.0917511,
                    1.0820790, 1.0820547, 1.0820914, 1.0820914, 1.0872870,
                    1.1095501, 1.0874169, 1.0918049, 1.0917946, 1.0915521,
                    1.0915344, 1.0873540, 1.0873439, 1.0957600, 1.0954438,
                    1.1107527, 1.1108221, 1.0884422, 1.0884422, 1.0884449,
                    1.0753182, 1.0677545, 1.1200000, 1.1200000, 1.1130731,
                    1.1137221, 1.1040600, 1.0892634, 1.0870684, 1.1044062,
                    1.1192730, 1.1192730, 1.0944584, 1.0965023, 1.1020563,
                    1.0958187, 1.0957928, 1.0952728, 1.1068136, 1.1068430,
                    1.1066686, 1.1163112, 1.1163046, 1.0929697, 1.1193987,
                    1.1197042, 1.0674001, 1.0739663, 1.0751689, 1.0819996,
                    1.1059027, 1.1094673, 1.1168377, 1.1195112, 1.0968612,
                    1.0726807, 1.1138587, 1.1138587, 1.0939653, 1.0942990,
                    1.1061418, 1.0735345, 1.1052742, 1.0976220, 1.1072170,
                    1.0795311, 1.1010086, 1.0955758, 1.1179142, 1.1179142,
                    1.1016272, 1.0874814, 1.1095058, 1.0902668, 1.0989082,
                    1.0952488, 1.0986718, 1.0988019, 1.0712597, 1.1020984,
                    1.1046546, 1.0881417, 1.0872848, 1.0871772, 1.0924635,
                    1.1196024, 1.1196024, 1.0895616, 1.0895595, 1.0868706,
                    1.1114100, 1.1063900, 1.1063900, 1.1156011, 1.1165152,
                    1.1051350, 1.0950981, 1.0878103, 1.0956564, 1.0985659,
                    1.0854818, 1.0498274, 1.1024586, 1.1024247, 1.0860897,
                    1.0845893, 1.0846300, 1.0749350, 1.1008841, 1.1009452,
                    1.0750276, 1.0992336, 1.0992169, 1.0946615, 1.0965792,
                    1.0967856, 1.1146966, 1.1171205, 1.1171205, 1.0989390,
                    1.0991612, 1.0696302, 1.0958252, 1.0999064, 1.1001885,
                    1.0792868, 1.0937964, 1.0937916, 1.0327590, 1.0828206,
                    1.0746027, 1.0711970, 1.0790559, 1.0901159, 1.0914616,
                    1.1083651, 1.1083651, 1.0957732, 1.0958614, 1.0920683,
                    1.1178978, 1.1178940, 1.0735331, 1.1007070, 1.0736697,
                    1.1171428, 1.0956104, 1.1095060, 1.0840549, 1.0848358,
                    1.0636159, 1.1143484, 1.0979343, 1.1173411, 1.1173904,
                    1.0924755, 1.1171181, 1.0939518, 1.0991066, 1.0885672,
                    1.1151573, 1.0830516, 1.1000282, 1.1133069, 1.1130824,
                    1.1130737, 1.0887497, 1.0908138, 1.0841411, 1.0717763,
                    1.0942196, 1.0968324, 1.0665734, 1.0683629, 1.0705187,
                    1.0846961, 1.1000543, 1.0678367, 1.0665648, 1.0827750,
                    1.0731343, 1.1069329, 1.1134768, 1.1080851, 1.1079355,
                    1.1127503, 1.1079274, 1.0389160, 1.0860092, 1.1162900,
                    1.0720509, 1.0720837, 1.0686670, 1.0688602, 1.1016161,
                    1.0678470, 1.0678145, 1.1200000, 1.0786866, 1.1057264,
                    1.1056923, 1.0732439, 1.0681052, 1.0749403, 1.0731632,
                    1.0732526, 1.0756104, 1.0756395, 1.0731084, 1.0691413,
                    1.0708757, 1.1044923, 1.0751754, 1.0727129, 1.0727321,
                    1.0686870, 1.0717767, 1.0806152, 1.0826709, 1.0939208,
                    1.0828062, 1.1107254, 1.1107234, 1.0829595, 1.0729180,
                    1.0910089, 1.0913813, 1.1065659, 1.1191240, 1.1191139,
                    1.0884936, 1.0875812, 1.0999083, 1.1022996, 1.1090701,
                    1.1169516, 1.0939072, 1.1147818, 1.1147818, 1.0773691,
                    1.0773691, 1.0871874, 1.0854348, 1.0886143, 1.0890340,
                    1.0754484, 1.1065824, 1.1154361, 1.1072515, 1.0876738,
                    1.0896085, 1.0957847, 1.1008298, 1.1116799, 1.1066809,
                    1.1173742, 1.1046956, 1.1178898, 0.9994085, 1.1015936,
                    1.0547882, 1.0845201, 1.0905854, 1.1171042, 1.1138323,
                    1.1103290, 1.0540432, 1.1132895, 1.0872553, 1.0937702,
                    1.0735230, 1.1069993, 1.1094523, 1.0995364, 1.0954211,
                    1.0951175, 1.0872432, 1.0947275, 1.1001500, 1.0855536,
                    1.0951482, 1.1086403, 1.0596797, 1.1154247, 1.0385877,
                    1.1015388, 1.0710590, 1.0978879, 1.0711500, 1.1075454,
                    1.1011632, 1.1175726, 1.1175626, 1.0875953, 1.0961606,
                    1.0981668, 1.0640212, 1.0754116, 1.0938650, 1.0966286,
                    1.1020252, 1.0990217, 1.1007697, 1.1024090, 1.1022440,
                    1.0911147, 1.0984246, 1.1018471, 1.1200000, 1.1151451,
                    1.0914839, 1.0914942, 1.1174631, 1.1174027, 1.1029815,
                    1.1188012, 1.1187945, 1.0753028, 1.0950846, 1.0950746,
                    1.0945934, 1.0935378, 1.0677907, 1.1189923, 1.1189923,
                    1.0690438, 1.0954563, 1.0713169, 1.0727256, 1.0169453,
                    1.0177809, 1.0897194, 1.0990056, 1.0693593, 1.0729884,
                    1.1083764, 1.0728551, 1.0789632, 1.0977020, 1.0963369,
                    1.0830651, 1.0830651, 1.1103025, 1.0916087, 1.1001944,
                    1.0729918, 1.0693599, 1.1024235, 1.1023327, 1.0932065,
                    1.0957727, 1.1130124, 1.0866386, 1.0679881, 1.0890326,
                    1.0979231, 1.0809555, 1.0808550, 1.0879120, 1.0959101,
                    1.1070542, 1.1040736, 1.0886396, 1.0937099, 1.0924466,
                    1.1196970, 1.0980001, 1.0586475, 1.0586256, 1.1059353,
                    1.0979134, 1.0434979, 1.0884039, 1.0835706, 1.0788752,
                    1.1190068, 1.0901210, 1.0848348, 1.1050577, 1.1037482,
                    1.1171128, 1.1005367, 1.1005563, 1.1183564, 1.1183564,
                    1.0869723, 1.0858133, 1.1012341, 1.1020199, 1.1173076,
                    1.0944504, 1.1159938, 1.0993660, 1.0907901, 1.1077040,
                    1.1185063, 1.0742941, 1.0796685, 1.0830868, 1.0662879,
                    1.0294638, 1.0827651, 1.0064805, 1.0897615, 1.0965510,
                    1.0743829, 1.0822718, 1.1096065, 1.1096065, 1.0939115,
                    1.1067673, 1.1034972, 1.0865794, 1.1197028, 1.1200000,
                    1.0377378, 1.0872371, 1.0969449, 1.0872176, 1.1009882,
                    1.0921300, 1.0919085, 1.1021270, 1.0766313, 1.1018002,
                    1.0439205, 1.0848235, 1.0982896, 1.1178944, 1.0708473,
                    1.1119533, 1.1124454, 1.1126144, 1.1125277, 1.1126513,
                    1.1108147, 1.1107868, 1.1133338, 1.1132230, 1.1012158,
                    1.0943610, 1.1187083, 1.1186944, 1.0788238, 1.1080991,
                    1.1080991, 1.0978085, 1.1142685, 1.0969629, 1.0954691,
                    1.1136226, 1.1066991, 1.1193557, 1.1193557, 1.1176397,
                    1.1177034, 1.0911289, 1.1066814, 1.1067571, 1.0582098,
                    1.1007776, 1.1007738, 1.0959571, 1.0981692, 1.0924764,
                    1.0924923, 1.0811205, 1.0917082, 1.0950926, 1.0942930,
                    1.0963824, 1.0880150, 1.0939464, 1.0925870, 1.0761188,
                    1.0450795, 1.0774576, 1.0391287, 1.0961360, 1.0972574,
                    1.0886611, 1.0955586, 1.1133733, 1.0943549, 1.0957841,
                    1.0885485, 1.0954413, 1.1166458, 1.0889960, 1.1068501,
                    1.1033692, 1.1134768, 1.0999041, 1.0690647, 1.0729629,
                    1.0880711, 1.0958813, 1.1009661, 1.0762154, 1.0940335,
                    1.0961238, 1.0707402, 1.0689196, 1.0746084, 1.0685464,
                    1.0931440, 1.0822670, 1.1019389, 1.0981150, 1.1013160,
                    1.0837548, 1.0996678, 1.0738740, 1.0898461, 1.0742805,
                    1.1179602, 1.1179602, 1.1200000, 1.1199710, 1.0990748,
                    1.1017135, 1.1038663, 1.0974889, 1.1089452, 1.1103038,
                    1.1099869, 1.1113604, 1.0988030, 1.0869956, 1.1191667,
                    1.0638727, 1.0942730, 1.0719992, 1.0801171, 1.1183775,
                    1.1183775, 1.0900618, 1.0916901, 1.0830533, 1.0746170,
                    1.1008042, 1.1040193, 1.0939490, 1.0759156, 1.1154416,
                    1.1174886, 1.0791570, 1.1104360, 1.1104181, 1.1162827,
                    1.0799778, 1.1106487, 1.1173244, 1.0692560, 1.0843917,
                    1.0839363, 1.0783881, 1.1143455, 1.1182582, 1.1191856,
                    1.1192275, 1.1111559, 1.0822820, 1.0802512, 1.0923159,
                    1.0922376, 1.0850912, 1.0829500, 1.0811850, 1.1139651,
                    1.0540782, 1.0556793, 1.0751069, 1.0928974, 1.0755215,
                    1.0751236, 1.1112427, 1.1144725, 1.0827745, 1.0746690,
                    1.0872538, 1.0874022, 1.1158890, 1.1149108, 1.1148837,
                    1.0668047, 1.0668070, 1.0660233, 1.0655664, 1.1188711,
                    1.1200000, 1.1132703, 1.0948943, 1.0963328, 1.0930075,
                    1.0917714, 1.0926948, 1.0894222, 1.0933659, 1.0870024,
                    1.0870150, 1.0869357, 1.1151919, 1.0666779, 1.0908290,
                    1.1048974, 1.0863721, 1.0867565, 1.0898670, 1.0899595,
                    1.0816054, 1.0941605, 1.1149494, 1.1192584, 1.1195763,
                    1.1195353, 1.0748035, 1.1151552, 1.0806017, 1.0976979,
                    1.1171286, 1.1199157, 1.1199800, 1.0765621, 1.0863428,
                    1.1076860, 1.0869092, 1.0601393, 1.0601440, 1.0543151,
                    1.1128364, 1.0711695, 1.0868726, 1.1030198, 1.0877289,
                    1.0917648, 1.1036101, 1.1105300, 1.1112795, 1.1126236,
                    1.1130463, 1.1115009, 1.0749448, 1.0738733, 1.0724941,
                    1.0733564, 1.0701039, 1.1182622, 1.0692394, 1.0927341,
                    1.0694398, 1.1111952, 1.1062882, 1.0871930, 1.0908731,
                    1.0932055, 1.0796825, 1.1098894, 1.1095372, 1.0899004,
                    1.0847638, 1.0943601, 1.0822908, 1.1029861, 1.0931632,
                    1.0706936, 1.0636131, 1.1144627, 1.0718387, 1.0859936,
                    1.0705966, 1.1156848, 1.0835467, 1.0942779, 1.0733645,
                    1.0755154, 1.0689302, 1.1127234, 1.0840273, 1.0778545,
                    1.0728592, 1.1148799, 1.0720855, 1.1169334, 1.0889187,
                    1.0888093, 1.0906511, 1.0907356, 1.0887212, 1.0887223,
                    1.0858146, 1.0919866, 1.0843936, 1.1139386, 1.0879931,
                    1.0814544, 1.0776319, 1.1122919, 1.1160915, 1.0847456,
                    1.0851626, 1.0855989, 1.0733987, 1.0733305, 1.0903537,
                    1.0866325, 1.1165374, 1.0902800, 1.0670112, 1.0670473,
                    1.0871308, 1.0913549, 1.0881253, 1.0677859, 1.1140379,
                    1.0933584, 1.0901757, 1.0846658, 1.0711635, 1.0735394,
                    1.1098142, 1.0797285, 1.0893984, 1.1116085, 1.1105156,
                    1.0710103, 1.0946343, 1.0963337, 1.0634727, 1.0864980,
                    1.0954974, 1.0707727, 1.1199805, 1.0874730, 1.0929894,
                    1.1021624, 1.1146416, 1.0893076, 1.0778096, 1.1172351,
                    1.0729643, 1.1155767, 1.0873466, 1.0890961, 1.1144813,
                    1.1054032, 1.0986338, 1.1179943, 1.1117826, 1.0838097,
                    1.0464854, 1.0572721, 1.1145157, 1.1132016, 1.1199814,
                    1.1200000, 1.1152421, 1.1152913, 1.1148477, 1.0874717,
                    1.0894800, 1.0623056, 1.1156769, 1.0907621, 1.0719282,
                    1.0868318, 1.0965683, 1.0878321, 1.0763292, 1.1115402,
                    1.1168981, 1.1138717, 1.0866609, 1.0898640, 1.0899770,
                    1.0871547, 1.0881231, 1.0882637, 1.0852076, 1.0828170,
                    1.0801006, 1.0876911, 1.0871076, 1.0845390, 1.1173899,
                    1.1073271, 1.0963337, 1.1026440, 1.1016521, 1.1107661,
                    1.0858468, 1.0902891, 1.0901834, 1.1059110, 1.0871795,
                    1.0895713, 1.0894997, 1.0867667, 1.0877610, 1.0846730,
                    1.1165472, 1.0877760, 1.1172708, 1.1173635, 1.0914015,
                    1.0925887, 1.0876769, 1.0877119, 1.0755853, 1.0661083,
                    1.0618729, 1.1141351, 1.1141923, 1.1054888, 1.1191315,
                    1.1191697, 1.1180009, 1.0311195, 1.0302489, 1.1004050,
                    1.0048345, 1.0930186, 1.1083136, 1.0208681, 1.1200000,
                    1.0673669, 1.0674574, 1.0661509, 1.0661668, 1.0510221,
                    1.0416225])
    Vm[np.where(bus_type==1)] = 1
    Va = np.zeros(n_bus)

    gen_bus_nr = np.array([17, 17, 26, 27, 27, 27, 26, 28, 28, 28, 
                        29, 29, 29, 42, 42, 42, 43, 43, 43, 44, 
                        44, 54, 55, 54, 56, 57, 57, 57, 58, 77, 
                        77, 78, 78, 78, 81, 81, 97, 97, 103, 104, 
                        105, 106, 115, 116, 117, 124, 125, 125, 126, 126, 
                        131, 131, 132, 132, 133, 133, 134, 145, 145, 146, 
                        147, 147, 147, 147, 149, 150, 150, 154, 154, 154, 
                        154, 155, 155, 155, 155, 157, 163, 163, 164, 164, 
                        179, 204, 204, 204, 205, 214, 214, 214, 214, 220, 
                        221, 239, 246, 309, 309, 309, 309, 309, 309, 309, 
                        310, 310, 311, 311, 313, 313, 314, 322, 365, 366, 
                        367, 368, 385, 385, 386, 443, 452, 452, 485, 536, 
                        536, 536, 536, 536, 537, 537, 537, 537, 551, 559, 
                        559, 559, 559, 559, 560, 560, 560, 560, 573, 626, 
                        626, 664, 665, 685, 685, 691, 716, 740, 763, 792, 
                        796, 796, 796, 796, 807, 808, 845, 845, 845, 845, 
                        877, 900, 900, 900, 975, 976, 980, 1001, 1002, 1006, 
                        1005, 1006, 1006, 1007, 1013, 1013, 1013, 1015, 1015, 1024, 
                        1025, 1026, 1113, 1114, 1113, 1115, 1152, 1152, 1153, 1153, 
                        1154, 1154, 1179, 1180, 1180, 1188, 1246, 1246, 1247, 1247, 
                        1247, 1248, 1248, 1286, 1287, 1288, 1342, 1343, 1343, 1351, 
                        1352, 1351, 1353, 1363, 1363, 1374, 1374, 1375, 1376, 1414, 
                        1415, 1415, 1432, 1432, 1433, 1433, 1435, 1434, 1439, 1461, 
                        1461, 1644, 1644, 1644, 1644, 1644, 1644, 1672, 1673, 1673, 
                        1674, 1684, 1685, 1685, 1685, 1685, 1684, 1776, 1793, 1793, 
                        1793, 1793, 1794, 1836, 1836, 1838, 1872, 1872, 1873, 1873, 
                        1906, 1907, 1926, 1926, 1928, 1929, 1944, 1947, 1957, 1962, 
                        1964, 1964, 1965, 1991, 1997, 1997, 1997, 2001, 2001, 2002, 
                        2007, 2014, 2014, 2014, 2015, 2015, 2035, 2039, 2041, 2046, 
                        2056, 2063, 2063, 2070, 2090, 2090, 2091, 2093, 2096, 2118, 
                        2140, 2175, 2202, 2203, 2208, 2214, 2215, 2228, 2235, 2253, 
                        2254, 2265, 2277, 2293, 2295, 2296, 2328, 2328, 2328, 2331, 
                        2343, 2354, 2354, 2355, 2360, 2366, 2370, 2431, 2431, 2438, 
                        2448, 2451, 2453, 2459, 2467, 2470, 2471, 2480, 2481, 2482, 
                        2483, 2485, 2488, 2491, 2497, 2500, 2509, 2511, 2523, 2526, 
                        2529, 2536, 2539, 2542, 2544, 2546, 2547, 2554, 2556, 2557, 
                        2563, 2564, 2568, 2571, 2573, 2574, 2578, 2582, 2589, 2606, 
                        2625, 2628, 2632, 2634, 2643, 2648, 2649, 2651, 2653, 2654, 
                        2655, 2658, 2662, 2667, 2668, 2674, 2675, 2681, 2684, 2690, 
                        2691, 2695, 2701, 2701, 2701, 2701, 2702, 2702, 2712, 2714, 
                        2716, 2719, 2720, 2720, 2720, 450, 1166, 1845, 2149, 17, 
                        845, 1006, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730])
    
    Pg = np.array([0.000,    0.000,  370.000,  360.000,  360.000,  370.000,
                370.000,  370.000,    0.000,  370.000,    0.000,  370.000,
                370.000,  130.000,    0.000,  110.000,  120.000,  130.000,
                0.000,  433.768,    0.000,  129.000,  129.000,    0.000,
                129.000,  129.000,    0.000,  129.000,   48.100,  225.000,
                225.000,    0.000,  225.000,  225.000,  380.000,    0.000,
                9.000,    6.000,  225.000,    0.000,  120.000,    0.000,
                0.000,  120.000,  110.000,  215.000,  225.000,  215.000,
                225.000,    0.000,  125.000,    0.000,    0.000,    0.000,
                0.000,  120.000,  120.000,    0.000,    0.000,    0.000,
                0.000,    0.000,    0.000,   90.000,   90.000,   90.000,
                0.000,  235.000,    0.000,  261.000,    0.000,    0.000,
                205.000,  206.000,    0.000,  206.000,    0.000,  200.000,
                200.000,  200.000,  170.000,  179.000,  179.000,  179.000,
                179.000,   35.000,    0.000,    0.000,   10.000,    1.550,
                1.550,    7.000,  370.000,    0.000,    0.000,   11.000,
                0.000,    0.000,    0.000,    0.000,   15.000,    0.000,
                15.000,    0.000,   30.000,    0.000,    0.000,    2.500,
                43.250,   43.250,   43.250,   43.250,  140.000,    0.000,
                10.000,    1.500,    5.210,    0.000,    1.400,   40.000,
                0.000,    0.000,    0.000,    0.000,   25.000,   25.000,
                0.000,    0.000,    1.400,   30.000,    0.000,    0.000,
                0.000,    0.000,   30.000,    0.000,    0.000,    0.000,
                1.000,   78.000,   30.000,  130.000,    0.000,  150.000,
                50.000,    2.430,    1.470,  129.000,   86.000,    1.000,
                68.100,    0.000,    0.000,    0.000,    0.000,   92.000,
                15.000,   15.000,    0.000,    0.000,   35.000,    0.000,
                50.000,    0.000,    0.000,  100.000,    0.000,  386.000,
                370.000,    6.000,    4.000,    4.000,    0.000,    0.000,
                15.000,    0.000,    0.000,    0.000,    0.000,   12.000,
                0.000,    0.000,    0.000,    0.000,   50.000,    0.000,
                17.000,    0.000,   17.000,    0.000,   16.000,    0.000,
                70.000,    0.000,    0.000,   50.000,   10.000,    0.000,
                0.000,    0.000,    0.000,    0.000,    0.000,    0.000,
                230.000,    0.000,   20.000,    0.000,   25.000,    0.000,
                120.000,  120.000,    0.000,    0.000,  125.000,  100.000,
                0.000,    0.000,    0.000,    0.000,    0.000,    0.000,
                10.000,   10.000,    0.000,    0.000,   10.000,   13.000,
                0.000,   46.000,    0.000,   15.000,    0.000,   15.000,
                15.000,    0.000,    0.000,  153.000,    0.000,  153.000,
                123.000,  110.000,    0.000,    0.000,    0.000,  110.000,
                0.000,    0.000,    0.000,   20.000,    0.000,    0.000,
                39.000,  225.000,    0.000,  225.000,    0.000,    0.000,
                0.000,    0.000,    0.000,    0.000,   10.000,   10.000,
                120.000,  120.000,    1.000,    1.200,    0.200,    0.100,
                17.000,    0.000,    0.000,   90.000,    0.000,   27.300,
                0.000,   40.000,   80.000,    0.000,    0.400,    0.000,
                0.000,   25.000,    0.000,    0.000,   14.000,    8.500,
                120.000,    0.100,    1.000,    0.000,    2.000,    0.200,
                0.000,   50.000,   28.000,   55.000,    3.000,    4.500,
                1.000,    0.800,    0.800,    0.200,    4.500,  200.000,
                200.000,    0.400,   14.500,   50.000,    0.000,    2.000,
                4.000,    1.000,    0.200,    0.200,    0.000,    0.000,
                2.000,    0.100,  190.183,    0.000,   72.500,    0.000,
                4.400,    2.500,    0.200,    3.000,    0.000,    0.500,
                0.600,   65.000,    0.100,    0.600,    0.300,   29.000,
                0.000,    0.000,    0.200,    0.300,    0.500,    0.000,
                4.000,   12.000,    3.500,   40.000,    0.000,    0.000,
                0.000,    4.000,   23.000,   10.000,    0.000,    0.000,
                0.000,    0.500,    1.000,   36.000,    0.000,    0.000,
                0.100,    0.000,    0.200,    5.000,    0.400,    0.000,
                0.000,    0.000,    0.500,    0.200,    0.100,    2.100,
                10.500,    0.200,    0.300,    0.000,    0.000,    0.000,
                0.700,    0.000,    0.000,    0.000,    0.000,    1.300,
                0.000,    0.000,   55.000,    2.500,    2.000,    0.000,
                0.000,    0.100,   25.000,   25.000,    0.000,    0.000,
                0.000,    0.000,   20.000,    0.000,    0.000,    5.800,
                0.000,    0.000,    0.000,    6.800,    0.700,    3.100,
                2.567,    0.690,    0.020,    0.020,    0.000,    0.000,
                0.000,    0.000,  125.000,  176.691,    0.000,  120.000])/Sbase
    
   
    Qg = np.array([0.0000000,    0.0000000,  -19.0000000,  174.1080000,  -19.0000000,
                 -17.3049000,  144.8930000,  -19.0000000,    0.0000000,  -19.0000000,
                 0.0000000,    8.9389600,  -19.0000000,  -34.0000000,    0.0000000,
                 -34.0000000,  -34.0000000,  -34.0000000,    0.0000000,  -47.0000000,
                 0.0000000,   -1.0000000,  -16.0000000,    0.0000000,  -16.0000000,
                 -16.0000000,    0.0000000,  -16.0000000,    0.0000000,   56.4466000,
                 56.4466000,    0.0000000,    0.0284002,    0.0284002,  -31.0000000,
                 0.0000000,    0.0000000,    0.0000000,  111.2280000,    0.0000000,
                 44.1891000,    0.0000000,    0.0000000,   -2.0000000,    0.0000000,
                 -35.0000000,   76.0784000,  -35.0000000,  -40.0000000,    0.0000000,
                 40.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 49.9848000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   94.0447000,  -23.0000000,
                 -48.0000000,    0.0000000,  -53.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,  -22.0000000,  -20.0000000,    0.0000000,
                 28.5083000,    0.0000000,   64.6061000,   10.9272000,  104.0000000,
                 65.6015000,  -60.0000000,  110.0000000,  -60.0000000,  -14.8355000,
                 0.0000000,    0.0000000,    0.0000000,   11.2737000,    0.7292800,
                 1.0000000,    6.0000000,   61.8297000,    0.0000000,    0.0000000,
                 7.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 15.0000000,    0.0000000,   15.0000000,    0.0000000,   20.0000000,
                 0.0000000,    0.0000000,    3.0000000,   12.0000000,   12.0000000,
                 8.7805000,   12.0000000,   40.9072000,    0.0000000,    7.0916000,
                 2.0000000,    2.0000000,    0.0000000,    0.7000000,   14.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,   17.5000000,
                 17.5000000,    0.0000000,    0.0000000,    0.7000000,    5.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,    5.0000000,
                 0.0000000,    0.0000000,    0.0000000,    4.0000000,    0.0000000,
                 0.0000000,   57.2913000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,  -16.0000000,    0.0000000,    0.3151720,
                 1.9118800,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 12.9886000,    0.0000000,   20.7457000,    0.0000000,    0.0000000,
                 33.7466000,    0.0000000,   19.2843000,   51.8210000,    1.4470000,
                 0.4470000,    0.4470000,    0.0000000,    0.0000000,    4.8000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,   18.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,   37.5000000,
                 0.0000000,    5.2510000,    0.0000000,    5.2510000,    0.0000000,
                 5.2500000,    0.0000000,   37.7826000,    0.0000000,    0.0000000,
                 0.0000000,    2.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,   75.6751000,
                 0.0000000,    4.0190000,    0.0000000,    5.0000000,    0.0000000,
                 18.1025000,   -2.0000000,    0.0000000,    0.0000000,   89.0000000,
                 40.7604000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    2.0000000,    2.0000000,    0.0000000,
                 0.0000000,    3.7500000,    6.7500000,    0.0000000,    3.8574500,
                 0.0000000,    1.9000000,    0.0000000,    1.9000000,    1.9340000,
                 0.0000000,    0.0000000,   44.3213000,    0.0000000,   22.1290000,
                 5.0062000,   40.0010000,    0.0000000,    0.0000000,    0.0000000,
                 26.2063000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,   26.0000000,   57.6425000,    0.0000000,
                 126.7850000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    2.4000000,    2.4000000,   11.2483000,
                 44.6899000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   13.1500000,    0.0000000,
                 8.0010000,    0.0000000,   22.5820000,   80.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   12.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   87.0210000,    0.0000000,
                 0.0000000,    0.0000000,    2.0000000,    0.0000000,    0.0000000,
                 12.0000000,    2.6550900,    2.2101100,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 30.7580000,   54.0848000,    0.0000000,    0.0000000,   15.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    2.0000000,    0.0000000,
                 42.2312000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    3.6220700,    0.0000000,    0.0000000,
                 0.2000000,    9.0995500,    0.0000000,    0.2000000,    0.0000000,
                 5.0000000,    0.0000000,    0.0000000,    1.4000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    3.0000000,    0.0000000,
                 5.0000000,    1.2000000,    0.0000000,    0.0000000,    1.5000000,
                 20.0000000,    1.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,   24.0000000,    1.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   10.0000000,    0.0000000,
                 1.3000000,    0.8000000,    2.5000000,    0.0000000,    0.1000000,
                 0.1000000,    0.0000000,    7.5000000,    0.1000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,    0.5000000,    0.0000000,
                 1.0000000,   35.0000000,    0.0000000,    1.0000000,    0.0000000,
                 0.0000000,    0.0000000,    2.5000000,    2.5000000,    0.0000000,
                 0.0000000,    0.0000000,    0.0000000,   14.0000000,    0.0000000,
                 1.2000000,    1.3000000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.1440000,    2.3500000,    0.1600000,    0.0010000,
                 0.0200000,   -0.1580000,    0.0000000,    0.0000000,    0.0000000,
                 0.0000000,    0.0000000,  -46.1050000,    0.0000000,  -10.0419000])/Sbase

    Pgen = np.zeros(n_bus)
    Qgen = np.zeros(n_bus)
    for ind,genbus in enumerate(gen_bus_nr):
        gen_bus = np.flatnonzero(bus_nrs == genbus)[0]
        Pgen[gen_bus] += Pg[ind]
        Qgen[gen_bus] += Qg[ind]

    Pld = np.array([0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,  61.358,  84.861,  49.000,
                  49.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  25.000,
                  26.000,  25.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,  13.000,  13.000,  13.000,
                  26.000,  89.067,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,  10.327,   8.032,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  27.000,
                  18.700,   4.934,   0.000,  18.000,   0.000,   0.000,   0.000,
                  44.864,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,  12.490,  17.327,
                  0.000,   0.000,   0.000,   0.000,  14.000,   0.000,   7.000,
                  0.000,   0.000,  73.779,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   8.000,   5.500,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,  11.000,  41.353,  14.000,
                  0.000,   0.000,   0.000,   0.000,   1.500,   0.000,  11.215,
                  10.333,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  25.553,
                  0.000,  11.000,  11.922,   0.000,   0.000,   0.000,  19.475,
                  16.000,   0.000,  10.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   5.540,  18.255,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   3.130,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   9.590,   4.795,   4.262,
                  6.500,   2.877,   3.197,  21.711,  12.254,   6.926,  10.655,
                  7.672,   8.524,   0.000,   6.606,   8.099,   0.746,   3.943,
                  3.197,  12.787,   4.475,   4.262,   2.451,   2.451,   5.327,
                  4.262,   9.057,  31.966,   5.860,   8.631,   5.967,   5.328,
                  0.000,  12.040,   3.836,   3.729,   3.729,   2.131,   0.000,
                  0.000,   7.458,   2.131,   2.131,   2.664,   3.516,  11.721,
                  11.721,  10.122,  12.787,   4.795,   3.197,   3.197,   7.992,
                  4.795,   8.098,   0.000,   1.279,  25.574,  34.097,  27.704,
                  1.066,   9.590,   5.967,   9.057,   6.393,   7.885,   0.426,
                  0.213,   0.000,  11.401,   0.000,   5.328,   2.131,   5.328,
                  5.328,   5.328,  14.918,  37.294,   1.386,   0.000,   8.525,
                  7.459,   6.926,   4.795,   3.729,   2.344,  11.402,   8.311,
                  4.902,   1.704,   4.582,   7.672,   4.262,   6.180,   8.205,
                  8.524,   0.000,   4.049,   7.992,   7.779,  13.746,  12.254,
                  14.147,  11.759,  15.573,   0.000,  10.354,   6.180,   8.525,
                  2.131,   4.794,   3.996,   6.127,   6.926,   8.205,  14.598,
                  12.787,  13.959,   5.967,   5.967,   7.459,  13.852,   9.057,
                  5.328,   3.197,  26.852,   7.459,   2.664,   5.860,   5.434,
                  7.352,   7.459,   9.590,   9.590,   2.664,   0.000,   0.000,
                  11.188,  11.188,   0.000,   3.729,   5.328,   6.926,  11.614,
                  7.672,   0.000,   4.262,   0.000,   1.066,   0.000,   0.000,
                  6.393,   6.926,   9.057,   0.000,   0.000,   5.861,   4.156,
                  53.277,  53.277,  53.277,  53.277,   0.000,   8.524,  17.048,
                  4.262,   0.000,  11.721,   2.664,   2.664,   6.393,   6.607,
                  5.860,   4.795,   2.131,  10.655,   7.459,  12.254,   0.000,
                  0.000,   2.131,   7.779,  11.934,   6.394,  12.360,   2.131,
                  0.000,   0.000,   3.729,   1.066,   4.262,   4.262,   0.000,
                  3.836,   2.065,   4.262,   3.197,  27.172,  22.909,   7.459,
                  6.393,   7.459,   5.754,  10.229,   6.713,   2.557,   0.000,
                  0.000,  11.188,   7.459,   6.926,  16.516,   3.197,   5.221,
                  14.918,  14.918,   5.328,   8.737,   8.098,   7.459,   8.524,
                  5.328,   9.910,   0.000,   7.672,   4.688,   6.926,   8.311,
                  5.541,   5.860,   4.262,  16.516,   0.000,   8.525,   2.131,
                  2.664,   4.263,   2.664,   2.770,  12.467,   4.795,   2.557,
                  6.393,   0.000,   4.476,   6.819,   3.197,   4.262,  14.918,
                  1.492,   9.590,   7.459,  10.016,   7.459,   4.262,  11.188,
                  6.393,   6.926,   7.459,  12.680,  10.656,   6.394,   6.180,
                  2.984,   4.156,   2.025,   3.197,  13.319,   1.066,   1.066,
                  7.992,   2.984,   2.344,  12.254,  16.729,   5.860,   2.131,
                  2.131,   4.795,   4.475,   5.328,   5.647,  19.180,   7.459,
                  3.197,   6.393,   3.729,  17.049,  17.049,  24.507,  23.762,
                  2.131,   6.926,   5.328,  14.917,  24.720,  12.254,  12.254,
                  14.066,   3.197,  26.639,  53.277,   2.131,   2.131,  19.925,
                  0.000,   2.131,   3.197,   7.459,   5.328,   2.131,   0.000,
                  18.115,  10.442,  10.442,   8.737,  13.853,  21.843,  21.844,
                  14.385,  38.359,  13.319,   3.197,   5.860,   5.860,  20.458,
                  20.458,  33.032,   3.943,  12.786,  15.127,  10.655,  10.655,
                  15.984,  21.844,  21.844,  21.311,  23.655,  38.359,  15.450,
                  9.590,   9.590,  28.769,  11.295,  36.229,   0.000,  10.656,
                  4.262,  11.188,   3.729,   0.000,   0.000,   6.062,   7.142,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   4.156,
                  2.132,   2.131,   3.197,  11.721,   6.393,  10.655,   8.631,
                  0.000,   4.795,   1.492,   1.279,   6.393,   0.213,   5.327,
                  9.803,   3.197,   2.131,   5.328,   7.459,   0.000,   4.347,
                  2.226,   6.151,  10.391,   4.898,  11.664,   0.000,  20.036,
                  4.771,   3.500,   9.436,   4.028,   4.771,   6.679,   9.967,
                  2.129,   2.439,  25.765,   9.542,   5.301,   5.673,   6.997,
                  14.168,  14.168,   6.256,   0.000,   0.000,   7.211,   3.181,
                  0.000,   0.000,   4.241,   5.301,   7.104,   3.971,   2.120,
                  6.892,   3.181,   0.000,   5.386,  17.535,   3.971,   3.758,
                  9.224,   6.910,   9.647,   3.181,   6.785,   8.482,   0.119,
                  4.559,   5.473,   4.241,   8.021,   7.953,   3.181,   8.482,
                  5.301,   2.439,   2.544,   3.605,   0.742,   2.544,   5.665,
                  9.861,   4.061,   1.255,  23.327,  11.664,   5.301,   1.130,
                  2.757,   0.530,   3.711,   8.059,   3.181,  13.000,   0.000,
                  5.832,  23.327,   6.574,   0.000,  12.724,  11.347,   9.516,
                  6.361,   2.120,  10.922,   6.361,   5.301,  19.086,   7.953,
                  17.535,   7.317,  14.208,   8.005,   7.528,   8.503,   0.000,
                  12.512,   4.666,   5.089,   4.531,   6.574,   6.510,  12.109,
                  6.361,   5.513,   0.000,   0.000,   2.544,   9.362,  10.052,
                  13.106,   0.000,  11.112,   5.587,  12.311,   5.301,   2.502,
                  2.757,  12.724,   8.377,   0.250,   4.241,   1.704,   0.000,
                  0.000,  36.581,  30.065,   5.301,  16.541,   3.074,   4.028,
                  14.102,   4.992,   2.120,   2.120,   5.727,   0.212,   2.120,
                  4.771,  36.326,   0.000,   1.255,  42.413,   0.000,   9.637,
                  3.181,   5.089,   6.256,   5.016,  13.000,   0.000,   4.241,
                  5.665,   4.241,   4.559,   6.892,   0.848,   3.393,   2.332,
                  7.104,  12.831,   7.422,   1.908,  14.168,   7.953,   7.422,
                  15.031,  29.689,   0.000,  10.709,   8.482,   1.255,   4.754,
                  25.388,   9.542,  10.604,   6.151,   5.301,   0.000,   7.422,
                  13.153,   6.361,   9.065,   9.224,   4.877,   8.800,   3.287,
                  0.000,   0.000,   3.181,  18.026,  15.905,   4.241,   0.954,
                  2.120,   0.000,   8.503,   4.771,  11.875,   6.361,   9.118,
                  7.135,   6.361,   2.968,   7.482,   0.000,   3.711,   4.771,
                  3.758,  10.391,   4.134,  14.844,   0.000,   0.250,   4.241,
                  11.664,  26.507,  24.744,   8.059,   1.130,   9.397,   4.973,
                  5.938,   4.241,   7.953,   8.800,   4.559,   2.650,   3.816,
                  1.166,   4.984,   9.542,   3.923,   7.528,   4.241,   5.832,
                  5.301,   9.542,  11.441,   0.000,   0.000,   8.588,   1.908,
                  3.181,   2.332,   6.574,   5.301,   2.120,   5.301,   0.000,
                  0.115,   0.000,   0.000,  39.012,   0.000,   8.043,   0.000,
                  1.838,   0.000,   8.735,  21.246,   0.000,   0.000,   8.500,
                  8.170,   0.000,   0.000,   5.165,   3.442,   2.412,   0.000,
                  12.622,   0.000,  29.209,  10.347,   0.000,   0.000,   5.750,
                  2.987,   5.985,   4.598,   2.181,   2.295,   0.000,   2.180,
                  1.377,   3.902,   9.412,   8.194,   4.256,   0.000,   4.019,
                  5.394,  10.327,   0.000,   2.529,   1.263,   0.000,   0.000,
                  0.000,   1.492,   3.104,   0.000,   8.039,   0.000,   5.858,
                  3.674,   0.000,  10.900,   3.000,   7.458,   0.000,   5.978,
                  7.238,   7.924,  12.424,   0.000,   8.611,   4.591,   1.606,
                  0.000,   0.000,   9.753,  16.064,  14.957,  12.744,   7.002,
                  0.574,   2.874,  17.899,   5.049,   3.789,   0.000,   0.000,
                  0.000,   0.000,   3.334,   6.893,   7.467,   7.700,   7.477,
                  6.884,   6.081,   7.463,   8.170,   0.000,   0.000,   0.654,
                  2.743,   4.133,   0.000,   5.737,   4.590,   0.000,   6.547,
                  4.478,   0.000,   3.562,   4.367,   0.000,   4.602,   0.000,
                  6.319,   8.609,   9.963,   5.672,   3.442,   0.000,   3.104,
                  0.000,   2.413,   0.000,   0.000,   1.836,   4.712,   0.000,
                  4.590,   0.000,   6.838,   0.000,   3.675,   4.940,   4.821,
                  4.361,  18.366,   0.000,   3.329,   6.774,  12.277,   0.000,
                  8.498,   0.000,   1.377,   0.000,   4.597,   0.000,   7.229,
                  4.016,   9.306,   4.707,   5.864,   0.000,   9.528,  13.436,
                  6.774,   4.252,  13.208,  12.633,  12.865,   7.458,  24.884,
                  24.884,   3.791,   2.986,  17.548,   0.000,   0.000,   0.000,
                  2.295,  10.616,   2.877,   0.000,   9.169,  30.999,   0.000,
                  1.725,   0.000,   4.590,   5.737,   3.681,   0.000,   0.000,
                  0.000,  35.022,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.688,
                  0.000,   0.229,   2.524,   0.000,   2.869,   0.574,   0.000,
                  0.000,  15.163,   0.000,   8.376,   6.770,  10.346,   8.505,
                  0.459,   0.000,   3.442,   0.000,   2.527,   0.000,   7.014,
                  0.000,   5.163,   3.901,   3.442,   6.770,   2.068,   8.517,
                  0.000,   1.377,   3.099,   0.459,   3.213,  11.474,   2.755,
                  3.214,   7.583,   6.893,   2.987,   5.403,   0.000,   0.000,
                  3.909,   1.837,   5.852,   0.000,   0.000,   0.000,   2.756,
                  2.181,   1.722,   1.606,   5.737,   0.000,   7.343,   2.295,
                  5.168,   0.000,   5.512,   2.644,   3.786,   0.000,   0.000,
                  0.000,   0.000,   5.163,   0.000,   7.233,   0.000,   0.000,
                  1.034,   1.608,   2.295,   0.000,   4.362,   4.134,   4.303,
                  0.000,   0.000,   3.675,   4.134,   7.455,   7.711,   8.835,
                  0.688,   0.000,   2.066,   0.000,   0.000,   2.641,   0.000,
                  0.000,   0.000,  13.111,   9.653,   5.625,   4.592,   2.871,
                  4.600,   2.639,  63.567,   0.000,  48.535,   0.000,   0.000,
                  0.000,   6.884,  36.717,   0.000,   0.574,   1.492,   2.869,
                  6.770,   1.836,   2.869,  22.277,  19.878,  21.336,   5.738,
                  2.869,  23.446,   2.180,  11.486,   1.033,  15.857,  10.453,
                  7.349,  11.027,  11.027,   0.000,   3.104,   0.000,   0.000,
                  4.245,   0.803,   0.000,   0.000,   1.836,   0.000,   0.000,
                  7.692,   5.166,   7.721,   1.377,   4.369,   0.000,   0.000,
                  0.000,  12.622,  10.327,  76.303,  99.816,   0.000,   7.477,
                  0.000,   0.918,   0.000,  19.620,   0.000,   0.000,   2.295,
                  0.688,   0.000,   0.000,   0.229,   0.803,   0.000,   3.334,
                  0.229,   2.869,   3.442,   0.000,   0.000,   4.594,   2.296,
                  1.378,   1.723,   6.111,   2.529,   1.147,   0.000,   4.247,
                  1.721,   0.574,   0.459,   1.378,  10.570,   4.142,   3.568,
                  5.057,   4.711,   0.000,   0.000,   5.511,   0.000,   0.000,
                  0.000,   8.727,  10.910,   1.606,   2.525,   0.000,   0.000,
                  2.296,   2.640,   0.000,   4.948,   4.476,   8.966,   9.939,
                  83.654,  29.193,   0.000,   2.295,   4.016,   0.000,   0.000,
                  0.000,   0.000,   2.983,   3.213,   4.021,   0.000,   0.803,
                  0.000,   0.000,   0.000,   1.147,   0.000,   3.332,   2.872,
                  12.851,   0.000,   3.557,   0.000,   7.019,   0.000,   3.213,
                  0.000,   6.199,   0.000,   5.742,   2.984,   7.343,   1.378,
                  7.808,   0.000,   4.016,   0.000,  10.240,  18.000,   0.000,
                  7.584,   9.082,   0.000,   5.065,   1.722,   1.493,   4.934,
                  7.114,   3.904,   7.008,   0.804,   6.118,   4.022,   0.574,
                  0.000,   0.000,   6.440,   3.330,   5.054,   7.606,   0.000,
                  0.000,   1.264,   1.954,   3.677,   4.136,   0.000,   0.115,
                  4.475,   0.000,   0.000,   0.000,   0.000,   0.000,   6.999,
                  6.770,   2.295,   0.000,   0.000,   0.000,   2.871,   1.952,
                  5.294,   0.000,   2.989,   2.068,   4.026,   4.597,   9.538,
                  0.000,   0.000,   6.196,   0.000,  22.020,  22.020,   5.173,
                  5.057,   0.000,   1.606,   7.812,   2.297,   0.000,  12.509,
                  8.918,   0.000,   4.257,   2.874,   5.742,  13.452,   0.000,
                  0.000,  11.504,   9.416,   0.000,  25.212,   4.478,   5.743,
                  1.149,   1.147,  15.045,  24.784,   0.000,   8.497,   0.000,
                  0.000,   8.627,   0.000,   3.675,   4.022,   2.527,   5.979,
                  0.000,   6.889,   0.000,   1.033,   1.378,   5.739,  12.632,
                  0.000,   0.000,   3.907,   1.262,   9.787,   8.643,   6.434,
                  8.390,   2.413,   0.000,  11.264,   8.272,   2.065,  16.982,
                  0.000,   3.557,  11.933,   3.676,   1.377,   0.000,   0.000,
                  5.163,   0.000,  11.262,   8.730,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   3.675,   0.229,   2.066,   2.984,
                  0.344,   0.688,   0.000,   0.000,   2.984,   9.659,   0.000,
                  7.125,   2.297,   0.000,   1.701,   0.000,  30.057,   3.662,
                  0.000,   9.537,  11.502,   3.910,  13.239,   0.000,   0.000,
                  6.439,   4.715,   0.000,   0.000,   0.000,   1.952,   0.000,
                  0.803,   3.558,   1.721,   8.045,   4.251,   0.000,   5.399,
                  6.657,   6.887,   6.672,   0.000,   2.965,   1.127,   6.095,
                  8.515,   0.000,   0.000,   5.978,   6.670,   0.000,   0.000,
                  0.000,   0.000,   0.000,  10.796,   2.640,   9.424,   0.000,
                  0.000,   0.688,   0.803,   5.513,   2.870,   4.021,   2.528,
                  2.525,   2.989,   6.094,   0.000,   8.072,   0.000,   0.000,
                  3.101,   2.296,   5.854,  12.630,  13.425,   5.163,   0.000,
                  1.722,   3.904,   2.986,  10.389,   0.000,   2.870,   6.658,
                  0.000,   5.737,   3.786,   2.298,   3.334,   3.217,   1.608,
                  13.803,   0.000,   0.000,   0.000,   5.858,   4.478,   0.000,
                  6.784,   5.754,   2.755,   0.000,   0.000,   0.000,   9.646,
                  2.760,   0.688,   1.263,   0.000,   0.000,   5.622,   0.000,
                  0.000,   5.056,   4.372,  14.733,   3.328,   0.000,   4.361,
                  6.414,   8.713,  19.506,  11.474,  10.346,   0.000,   2.068,
                  0.000,   1.033,   6.435,   0.000,   0.000,   2.872,   0.000,
                  1.263,   1.953,   5.518,   4.942,   5.622,   0.000,   0.918,
                  0.688,   8.045,   8.039,   0.000,   0.000,   8.284,   1.608,
                  0.000,   0.000,   8.841,  11.023,   9.211,   0.000,   0.000,
                  7.940,   5.737,   4.475,   1.147,   1.836,   4.018,   1.033,
                  4.016,   8.032,   3.676,   0.000,   5.979,   3.215,   2.066,
                  7.700,   0.000,   3.100,   7.363,   2.871,   1.721,   0.000,
                  6.888,   0.000,   6.439,   0.000,   3.445,   9.073,   5.401,
                  6.089,   9.886,   3.215,   2.760,   0.000,   3.099,   0.000,
                  4.704,   0.000,   0.000,   4.384,   1.726,   0.000,   1.377,
                  0.918,   5.513,   0.000,   0.000,   0.000,   3.220,   3.796,
                  15.082,   0.000,   5.749,   4.597,   0.000,   0.000,   0.000,
                  2.295,   0.000,  10.694,  11.154,   9.668,   8.860,   0.000,
                  0.000,   9.771,  10.331,   0.803,   0.000,  12.067,   0.000,
                  0.000,   0.229,   0.000,   3.213,   3.328,   0.918,   0.000,
                  0.000,   2.182,   2.180,   5.741,   4.131,   0.000,   5.737,
                  9.179,   3.451,   4.021,   2.066,   4.704,   4.934,   0.000,
                  0.000,   1.493,   0.229,   0.000,   9.187,  16.951,  15.000,
                  14.819,   6.434,   4.366,   0.000,   3.910,   4.488,   3.445,
                  0.000,   0.000,   4.590,  11.510,   4.574,   2.184,   5.287,
                  2.298,   0.000,   0.000,   0.000,   1.836,   2.524,   5.634,
                  10.361,   2.295,   0.000,   1.492,   3.443,   0.000,   0.000,
                  9.294,   3.442,   0.000,   5.744,   4.136,   4.360,   0.000,
                  5.400,   4.820,   0.000,   0.000,   9.203,   1.033,   0.000,
                  1.837,   0.000,   7.688,   0.000,   0.000,   2.066,   4.827,
                  0.000,   0.000,   8.285,   7.355,   7.924,   3.902,   0.000,
                  1.263,   1.492,   9.994,   1.147,   0.000,  11.016,  16.665,
                  1.721,   5.217,   9.590,   4.597,   8.611,  10.327,   8.032,
                  1.149,   0.803,   2.640,   3.443,   5.859,   1.493,   6.901,
                  6.325,   0.000,   8.627,   5.975,   2.525,   7.114,   0.000,
                  3.100,   8.163,  12.440,   0.000,   3.222,   0.000,   2.870,
                  0.000,   2.068,   3.334,   3.447,   0.000,   7.917,   0.000,
                  4.017,   5.167,   9.198,   7.011,  21.800,   0.000,   7.458,
                  0.000,   4.823,   0.000,   4.593,   5.166,   0.000,   0.000,
                  5.747,   0.000,   0.803,   0.000,   0.000,   0.000,   7.458,
                  9.169,  13.404,   5.404,   7.931,   7.941,   3.217,   4.252,
                  0.000,   4.361,   1.836,   5.057,   4.707,   0.000,   5.749,
                  2.527,   0.000,   8.614,  11.028,   6.559,   8.959,   5.747,
                  0.000,   0.000,   0.000,   0.000,   0.000,  11.409,   6.554,
                  6.431,   0.000,  10.233,  13.346,   4.022,   3.676,   3.677,
                  8.957,   0.000,   0.000,  13.030,   4.821,   7.242,   0.000,
                  3.098,  13.000,   0.000,  13.000,   0.000,   2.527,   6.903,
                  6.552,  14.367,   0.000,   0.000,   0.000,   0.000,   4.138,
                  9.195,   9.771,   3.676,   2.644,   7.233,   2.986,   0.000,
                  7.471,   0.000,  10.220,   1.147,   2.066,   4.828,   6.435,
                  4.592,   0.115,   0.000,   0.688,   1.147,   0.000,   0.000,
                  0.000,   0.000,  11.528,   0.000,   2.877,   2.297,   0.000,
                  0.000,   2.065,   6.319,  10.462,   7.814,  13.884,  10.671,
                  4.483,   9.210,   6.671,   0.000,   0.000,   4.711,   1.147,
                  6.655,   0.000,   0.000,   2.873,   1.377,   5.508,   3.672,
                  1.377,   0.000,   5.967,   2.524,   9.437,   0.000,   1.378,
                  0.000,   6.885,  19.220,   0.000,   0.000,   1.837,   2.296,
                  0.000,   2.410,   6.087,   4.025,   7.249,   1.722,   0.000,
                  2.411,   8.956,   3.905,   3.559,   0.918,   0.000,   0.000,
                  11.933,  11.605,   4.125,   7.530,  15.357,  16.246,  12.954,
                  7.229,   5.451,   0.000,   7.363,   0.000,   4.864,  20.943,
                  13.706,   6.330,   0.000,   5.599,   5.703,   5.161,   5.449,
                  11.301,   3.684,  11.188,   4.420,   3.684,   7.360,   4.715,
                  1.035,   7.222,   0.705,   4.706,   4.420,   0.000,   7.950,
                  4.021,   6.039,   0.000,   5.743,   0.000,   2.934,  14.281,
                  5.161,   4.431,   6.194,   2.653,   6.336,   1.178,   0.000,
                  0.000,   0.000,   0.000,   2.503,   5.307,  13.544,   1.653,
                  7.656,   2.295,  10.600,  10.911,   3.449,   4.954,   3.243,
                  5.893,   7.235,  17.022,   0.000,   2.574,   2.141,   9.568,
                  2.949,   0.265,   0.000,   6.339,   5.601,   5.160,  13.250,
                  6.190,   0.000,   5.157,   4.419,   4.917,   5.153,   0.000,
                  8.097,  11.050,  10.167,   6.477,  33.273,   2.355,   6.931,
                  4.127,   4.422,   6.941,  11.337,   4.426,  11.799,   3.389,
                  0.000,   8.843,   6.190,  25.912,   4.407,  16.230,   2.949,
                  3.243,   0.300,   0.000,   4.565,  22.929,   0.042,  36.941,
                  36.806,  28.988,   0.650, 147.162,   0.000,   0.737,   3.243,
                  9.568,   6.327,   2.211,   9.005,  10.319,   6.630,   3.828,
                  12.073,  30.391,  11.249,   7.964,   3.488,   0.000,   5.455,
                  4.422,   4.130,   8.835,   5.304,  10.315,  10.676,  14.593,
                  0.000,  13.840,  11.790,  11.496,   4.706,   6.193,  15.753,
                  7.081,   7.067,   5.424,   0.000,   0.000,   0.000,  22.589,
                  23.633,  10.166,   6.924,  12.388,   5.428,   5.899,  10.778,
                  13.899,   0.000,   9.726,   0.000,   4.029,   0.000,   2.241,
                  2.508,   7.073,  27.916,  22.147,  22.145,  14.723,   0.737,
                  4.424,   3.608,   0.000,   1.326,   0.000,   0.000,  14.005,
                  8.845,   5.161,   0.294,   0.294,  17.711,   3.279,   2.503,
                  0.000,  16.207,   7.970,  40.491,   1.620,   5.412,   7.664,
                  7.888,   5.899,   5.602,  11.483,   9.584,   0.000,   3.687,
                  10.920,   7.233,   2.062,   8.264,   4.718,   0.000,   5.159,
                  4.418,  11.206,   0.000,   2.503,   2.792,   2.212,  11.795,
                  4.419,   0.000,  16.635,   0.000,  14.593,  13.644,   0.000,
                  10.904,  12.375,   2.608,   0.000,   0.000,   1.030,   1.473,
                  5.744,   4.566,   8.424,   4.127,   3.682,   5.758,   6.335,
                  6.330,   3.831,   1.032,   3.986,   9.138,   1.767,   1.915,
                  4.420,   6.626,   1.178,   9.284,   3.426,   2.141,   9.733,
                  4.274,  10.748,   3.094,   5.159,   8.991,   5.888,   2.502,
                  8.097,  10.893,   6.784,   7.214,   2.933,  15.897,   3.389,
                  5.011,   8.401,   7.428,  12.692,   5.307,   0.000,   5.898,
                  3.532,   6.040,  10.462,   2.319,  19.021,  16.943,   5.598,
                  4.715,   7.958,  22.344,   0.000,   0.000,  13.715,   0.000,
                  0.000,   4.642,  19.025,   5.894,   2.458,   1.326,   4.722,
                  8.254,   4.124,   5.307,   6.626,   3.827,   4.592,   2.946,
                  6.927,  14.411,   6.038,   0.000,  10.023,  11.924,   0.000,
                  0.000,  41.397,  19.737,   0.000,   0.000,  11.814,   2.906,
                  4.718,   9.128,   1.915,  12.385,  12.367,   8.981,   0.000,
                  3.682,   5.743,  11.779,   0.000,   0.000,   5.255,   6.036,
                  9.422,  11.487,   3.975,   7.214,  15.753,   6.626,   7.067,
                  10.317,   0.000,   1.769,  18.622,   8.981,   2.652,   9.579,
                  6.330,   5.449,   1.326,   6.640,   7.518,   7.664,   2.357,
                  4.577,  14.267,   0.000,   9.863,  60.656,   2.948,   2.652,
                  6.483,   6.931,   6.103,   4.574,   2.796,   0.000,   4.127,
                  0.000,   0.000,   7.360,  14.843,  12.846,   3.383,   5.461,
                  4.420,   5.752,  20.348,   9.287,   4.420,  13.860,   8.981,
                  15.178,  11.810,  14.281,  10.600,  16.194,  12.683,   1.473,
                  4.716,   0.000,  10.321,   0.000,   6.784,   1.473,   6.044,
                  17.373,   0.000,   1.923,  13.857,   0.000,   3.682,   0.737,
                  11.206,   6.204,   5.011,   9.422,   5.751,   4.565,   6.631,
                  20.170,   8.409,  11.369,   6.036,  16.217,   2.946,   0.883,
                  3.240,   2.355,   3.532,   7.950,  20.447,   9.873,  21.822,
                  8.843,   2.061,   8.108,  13.985,   8.097,   0.000,  10.307,
                  1.471,   4.270,   6.330,   1.030,   3.243,   0.000,   0.000,
                  12.769,   8.853,   4.573,   0.000,   1.178,  10.307,   5.235,
                  1.427,   5.898,   6.331,   4.853,   6.043,  11.778,   8.689,
                  8.837,   4.718,   4.422,   9.725,   4.418,   6.779,   6.036,
                  12.662,  17.208,   3.094,   8.247,  13.258,   8.244,   8.244,
                  7.508,   9.568,   6.626,  13.110,   9.422,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                  5.165,   2.756,   7.461,   8.106,  52.502,   7.375,   0.000,
                  7.371,   0.000,  12.373,   5.898,  27.384,  23.568,   6.353,
                  11.943,   0.863,   9.873,   3.386,   3.978,  10.174,  10.025,
                  5.601,  11.462,   3.243,   5.157,   0.000,   5.898,   0.000,
                  0.000,   0.000,   4.865,   0.737,  10.207,   8.007,   6.625,
                  8.124,   8.007,   9.007,   0.000,   7.177,  10.063,   6.314,
                  14.906,   5.304,   4.504,   7.811,   3.864,  16.563,   8.281,
                  0.000,   0.000,   8.507,   2.946,   7.177,   9.471,   4.600,
                  15.321,  18.325,   6.625,   6.760,   6.625,   3.683,   9.508,
                  10.734,   3.803,   3.002,   3.503,   0.500,   5.505,   3.303,
                  0.601,  11.717,   4.003,   0.000,   0.000,   8.207,   7.006,
                  9.208,   6.405,   9.386,   0.000,   2.002,   0.000,   0.000,
                  7.198,   7.051,   0.000,   0.000,  10.313,   4.525,   3.999,
                  4.504,   7.510,   7.729,   5.407,   3.503,  11.576,  14.312,
                  0.000,   0.300,   2.002,   0.421,   8.711,   0.000,   0.000,
                  0.000,   3.503,   5.505,  14.512,   7.360,  15.459,   6.073,
                  6.625,  33.125,   5.505,   3.503,   6.608,   1.104,   1.546,
                  18.205,   0.000,   9.997,   8.629,  10.008,   9.997,   4.104,
                  9.208,   7.729,  10.008,   5.079,  11.510,  15.513,   3.313,
                  3.002,   7.729,  11.510,   7.106,   9.937, 132.710,   5.407,
                  6.405,   7.506,   3.455,   8.411,   0.000,   9.763,  10.008,
                  10.008,   7.006,   0.315,  14.420,   4.946,   6.505,   4.056,
                  10.524,  15.459,   7.806,   8.007,   6.405,  10.839,   3.002,
                  0.902,   6.294,   7.998,  11.510,  13.011,   9.386,   9.386,
                  7.006,  12.017,   4.504,   1.988,   3.754,   2.002,   5.366,
                  13.890,   0.000,   0.000,  11.009,   5.770,   7.006,  14.417,
                  12.010,   5.893,   8.007,  16.514,   1.801,   2.002,  13.250,
                  5.004,   9.508,  18.015,   5.852,   4.417,  11.892,   0.149,
                  3.002,  12.768,   7.606,   0.000,   2.105,   3.473,   5.261,
                  2.302,   4.704,   1.683,   9.050,   3.604,   4.195,   1.201,
                  2.209,   0.149,  10.839,   4.858,   6.073,  12.146,  27.361,
                  0.000,   7.506,  13.049,  16.943,   4.003,   0.000,  18.025,
                  7.998,   2.981,  13.250,   5.788,  10.514,   4.205,  19.875,
                  8.061,  33.027,  15.012,   0.000,  10.008,   6.514,  14.012,
                  5.505,   2.002,   7.893,   5.683,   5.505,   5.505,   0.000,
                  0.000,   7.506,   7.006,   5.407,  15.154,   0.105,  21.889,
                  7.006,   0.105,  18.015,   5.261,   5.521,   5.051,   4.604,
                  91.646,   3.305,   3.644,  16.100,  10.514,   0.000,   8.419,
                  10.365,   7.811,   5.858,  11.115,   8.281,   9.463,   9.386,
                  3.305,  11.510,   6.505,  10.814,  21.046,  13.260,   8.507,
                  14.417,   6.308,  16.372,   6.073,  13.969,   9.798,   0.000,
                  14.870,   4.315,   0.000,   0.000,   0.000,  36.437,   3.313,
                  5.473,   6.314,  70.058,  50.041,   6.005,  13.611,  12.698,
                  0.000,   0.000,  12.698,   0.000,   0.000,   1.501, 177.604,
                  431.500, 371.200, 136.331,   0.000,   0.000, 307.582,   0.000,
                  17.000,   0.000,  33.000,  16.000,  25.000,  65.000])/Sbase
    
    Qld = np.array([0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,  72.054, 122.060,  74.000,
                    74.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,  40.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   3.714,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  70.000,
                    42.000,   0.817,   0.000,  57.000,   0.000,   0.000,   0.000,
                    -6.400,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   3.696,   6.126,
                    0.000,   0.000,   0.000,   0.000,  40.000,   0.000,  18.000,
                    0.000,  13.892,   8.097,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,  15.000,  10.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,  17.500,  61.200,  33.000,
                    0.000,   0.000,   0.000,   0.000,   9.000,   0.000,   8.300,
                    7.700,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,  12.900,
                    0.000,  12.000,  13.700,   0.000,   0.000,   0.000,  64.300,
                    33.000,   0.000,  25.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   2.930,  11.500,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   1.500,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   1.000,   0.500,   2.000,
                    5.400,   0.600,   0.300,  12.300,   2.000,   2.000,   1.500,
                    3.000,   2.000,   0.000,   2.100,   1.500,   0.100,   0.200,
                    0.500,   1.000,   0.700,   0.700,   0.980,   0.980,   1.000,
                    1.000,   0.400,  28.001,   0.800,   2.940,   0.200,   0.500,
                    0.000,   1.000,   1.180,   0.400,   0.400,   1.100,   0.000,
                    5.000,   1.000,   0.000,   0.000,   0.500,   0.100,   5.400,
                    4.000,   1.000,   1.500,   0.000,   0.650,   0.650,   2.470,
                    0.000,   1.500,   0.000,   1.000,  22.000,  30.000,   0.800,
                    0.000,   0.800,   2.000,   1.500,   0.500,   1.850,   0.100,
                    0.060,   0.000,   5.180,   0.000,   1.400,   0.500,   0.500,
                    1.000,   1.000,  12.000,  33.000,   0.400,   0.000,   2.900,
                    2.000,   0.500,   0.000,   2.500,   0.200,   4.100,   1.500,
                    1.510,   1.400,   0.500,   0.800,   0.700,   2.110,   2.530,
                    0.500,   0.000,   1.500,   2.720,   2.400,   4.240,   3.780,
                    9.046,   7.000,   8.300,   0.000,   5.480,   5.150,   1.200,
                    0.500,   1.500,   1.250,   1.250,   2.500,   2.530,   4.510,
                    1.500,   4.760,   1.840,   1.840,   1.000,   3.000,   2.800,
                    2.000,   1.000,   8.290,   1.000,   0.800,   0.800,   0.200,
                    2.270,   2.700,   2.700,   2.700,   0.500,   0.000,   0.000,
                    4.000,   3.500,   0.000,   0.500,   1.000,   0.400,   3.590,
                    1.800,   0.000,   1.000,   0.000,   0.300,   0.000,   0.000,
                    1.900,   2.500,   1.200,   0.000,   0.000,   0.800,   0.800,
                    10.000,  10.000,  10.000,  10.000,   0.000,   0.800,   1.000,
                    1.000,   0.000,   3.100,   0.700,   0.700,   1.500,   0.400,
                    1.200,   0.800,   1.000,   2.100,   1.800,   3.500,   0.000,
                    0.000,   0.500,   2.800,   3.680,   2.000,   4.940,   0.730,
                    0.000,   0.000,   0.300,   0.000,   1.200,   1.000,   0.000,
                    1.000,   0.920,   1.000,   0.300,   7.000,   6.601,   1.000,
                    1.000,   1.000,   1.400,   0.500,   0.800,   0.500,   0.000,
                    0.000,   1.500,   0.200,   0.800,   2.000,   0.800,   0.800,
                    4.200,   4.200,   1.000,   2.700,   1.700,   0.499,   0.300,
                    2.000,   3.060,   0.000,   1.200,   0.300,   2.200,   1.000,
                    1.710,   2.000,   1.700,   3.000,   0.000,   0.500,   0.300,
                    0.400,   1.500,   0.300,   0.940,   3.850,   0.500,   0.800,
                    1.800,   0.000,   0.000,   0.400,   0.500,   1.500,   4.000,
                    0.200,   2.400,   1.700,   1.800,   0.401,   1.000,   1.500,
                    3.000,   1.500,   1.600,   5.400,   1.200,   0.500,   1.910,
                    1.000,   1.280,   0.100,   0.500,   1.000,   0.500,   0.500,
                    1.500,   0.200,   0.500,   5.000,   5.170,   1.000,   0.500,
                    0.000,   0.600,   0.600,   1.200,   1.300,   7.000,   4.000,
                    0.900,   1.800,   0.000,   4.800,   4.800,   6.900,   5.580,
                    1.000,   2.000,   0.500,   4.200,   6.960,   3.450,   3.450,
                    3.300,   1.000,   7.500,  15.000,   0.501,   0.500,   6.790,
                    0.000,   1.000,   1.000,   1.750,   1.250,   0.600,   0.000,
                    4.250,   2.450,   2.450,   1.600,   1.400,   5.130,   5.130,
                    3.380,  10.800,   3.749,   0.500,   1.380,   1.380,   5.760,
                    5.760,   9.299,   0.200,   5.677,   7.300,   3.000,   3.000,
                    3.750,   5.130,   5.130,   5.000,   6.660,  10.800,   4.350,
                    2.700,   2.700,   6.750,   3.180,  10.200,   0.000,   4.000,
                    0.500,   0.500,   1.000,   0.000,   0.000,   3.000,   4.500,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.900,
                    1.500,   0.500,   0.000,   0.500,   0.600,   2.300,   3.450,
                    0.000,   0.900,   0.400,   0.300,   1.800,   0.060,   2.000,
                    0.900,   0.990,   1.000,   0.400,   2.000,   0.000,   1.120,
                    1.000,   1.000,   2.400,   0.500,   1.500,   0.000,   9.512,
                    1.000,   2.000,   2.500,   0.800,   0.000,   0.900,   1.400,
                    1.700,   0.300,   4.300,   1.500,   0.000,   2.600,   1.800,
                    1.293,   1.293,   0.400,   0.000,   0.000,   0.400,   0.500,
                    0.000,   0.000,   0.000,   0.200,   1.800,   0.340,   0.000,
                    1.947,   0.000,  -0.002,   2.581,   7.911,   0.340,   2.800,
                    2.900,   2.246,   0.300,   0.900,   1.100,   2.000,   0.080,
                    0.200,   0.000,   1.000,   2.691,   0.000,   0.000,   0.000,
                    1.000,   0.700,   0.000,   0.400,   0.000,   0.600,   0.340,
                    2.400,   0.300,   0.910,   4.500,   1.000,   0.000,   0.068,
                    0.100,   0.000,   0.600,   0.100,   0.000,   0.000,   0.000,
                    0.500,   4.500,   1.200,   0.000,   2.600,   1.200,   0.272,
                    0.500,   0.500,   1.000,   0.000,   2.000,   2.000,   1.500,
                    6.272,   3.000,   2.800,   2.100,   0.430,   0.817,   0.000,
                    2.800,   1.600,   1.000,   0.680,   2.800,   1.050,   0.870,
                    1.000,   1.500,   0.000,   0.000,   0.900,   1.240,   1.720,
                    0.970,   0.000,   1.260,   0.210,   3.950,   0.000,   1.800,
                    0.000,   2.000,   1.100,   0.180,   1.000,   0.000,   0.000,
                    0.000,  12.800,  11.166,   1.000,   3.600,   0.700,   0.700,
                    3.300,   1.000,   1.000,   0.500,   0.000,   0.000,   0.500,
                    0.000,  16.476,   0.000,   1.100,  10.000,   0.000,   0.953,
                    0.000,   0.000,   0.900,   4.030,   0.000,   0.000,   0.500,
                    0.000,   1.000,   0.000,   0.300,   0.500,   0.100,   0.700,
                    2.000,   2.100,   0.000,   1.200,   0.545,   1.000,   0.300,
                    9.100,   5.000,   0.000,   1.100,   2.000,   0.710,   0.400,
                    1.701,   3.000,   0.000,   1.200,   1.000,   0.000,   3.000,
                    6.618,   1.000,   0.340,   1.800,   0.600,   0.000,   0.300,
                    0.000,   0.000,   1.000,   1.900,   3.000,   0.200,   0.200,
                    0.500,   0.000,   1.293,   1.000,   3.600,   1.500,   2.100,
                    6.125,   0.500,   0.100,   0.000,   0.000,   0.500,   0.000,
                    1.447,   1.900,   0.300,   3.500,   0.000,   0.100,   0.500,
                    3.500,   7.000,   7.000,   2.200,   0.000,   4.058,   1.820,
                    0.700,   0.200,   1.300,   1.400,   1.000,   1.000,   0.400,
                    0.200,   1.100,   1.500,   0.700,   1.000,   1.000,   0.200,
                    0.000,   1.000,   0.880,   0.000,   0.000,   1.700,   0.600,
                    0.300,   1.600,   0.500,   0.600,   0.000,   1.000,   0.000,
                    0.000,   0.074,   0.000,  18.316,   0.000,   2.644,   0.000,
                    0.430,   0.000,   1.913,   5.731,   0.000,   0.000,   0.968,
                    2.206,   0.000,   0.000,   3.067,   1.486,   0.680,   0.000,
                    2.971,   0.083,   5.988,   2.597,   0.000,   0.000,   1.846,
                    0.842,  -1.614,  -1.286,   0.517,   0.000,   0.000,   2.006,
                    0.446,   2.918,   2.109,   2.991,   1.337,   0.000,   0.985,
                    1.367,   2.155,   0.000,   0.904,   0.513,   0.000,   0.000,
                    0.000,   0.223,   0.777,   0.000,   2.711,   0.000,   1.111,
                    0.220,   0.000,   2.971,   5.000,   1.486,   0.000,   1.715,
                    1.611,   1.820,   0.975,   0.000,   1.597,   0.854,   0.149,
                    0.000,   0.000,   1.486,   4.457,   3.863,  -0.208,   1.970,
                    0.209,   0.839,   5.571,   1.783,   0.814,   0.000,   0.000,
                    0.000,   0.000,   0.484,   1.951,   2.322,   2.327,  -1.212,
                    -0.900,   1.411,   1.653,   2.321,   0.000,   0.000,   0.224,
                    0.898,   1.334,   0.000,   1.114,   0.743,   0.000,   1.928,
                    0.963,   0.000,   0.857,   1.563,   0.000,   0.919,   0.000,
                    1.823,   1.441,   1.909,  -2.004,   0.000,   0.000,   0.772,
                    0.000,   0.222,   0.000,   0.000,   0.297,   1.282,   0.000,
                    3.714,   0.000,   1.258,   0.000,   0.314,   0.619,   1.077,
                    1.226,   8.204,   4.000,   1.003,   1.894,   6.261,   0.000,
                    2.266,   0.000,   0.446,   0.000,   1.271,   0.000,   0.966,
                    1.337,   3.064,   1.653,   1.698,   0.000,   1.521,   1.099,
                    2.266,   1.491,   3.313,   2.918,   4.795,   1.486,  53.714,
                    52.972,   0.562,   0.169,   6.780,   0.000,   0.000,   0.000,
                    0.607,   3.662,   1.762,   0.000,   2.961,   7.137,   0.000,
                    0.732,   0.000,   1.114,   1.114,   0.863,   0.000,   0.149,
                    0.000,  15.010,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.136,
                    0.000,   0.000,   0.966,   0.000,   0.371,   0.223,   0.000,
                    0.000,   1.820,   0.000,   1.411,   1.560,   0.987,   1.209,
                    0.134,   0.059,   0.446,   0.111,   0.761,   0.000,   0.424,
                    0.000,   1.560,   0.520,   3.640,   1.263,   0.513,   2.718,
                    0.000,   0.223,   0.761,   0.113,   0.446,   1.189,   0.854,
                    0.929,   1.504,   1.385,   0.767,   1.618,   0.000,   0.000,
                    2.082,   0.147,   0.371,   0.000,   0.000,   0.000,   0.311,
                    0.375,   0.828,   0.744,   0.743,   0.000,   0.297,   0.817,
                    1.780,   0.000,   1.260,   0.681,   0.817,   0.000,   0.000,
                    0.000,   0.000,   0.669,   0.000,   1.597,   0.000,   0.000,
                    0.493,  -0.327,   0.594,   0.000,  -0.721,  -0.807,   8.120,
                    0.000,   0.000,   1.652,   0.842,   1.843,   1.097,   5.571,
                    0.000,   0.000,   0.233,   0.000,   0.000,   0.759,   0.000,
                    0.000,   0.000,   4.080,   2.022,   2.729,   2.644,   0.464,
                    2.570,   0.669,  21.246,   0.000,  13.743,   0.000,   0.000,
                    0.000,   1.857,   1.114,   0.000,   0.223,   0.297,   0.000,
                    0.446,   0.594,   0.669,   4.929,   4.431,   4.716,   1.546,
                    0.773,   8.978,   1.430,   2.592,   0.363,   3.558,   2.714,
                    1.951,   2.518,   2.518,   0.000,   0.769,   0.000,   0.000,
                    0.891,   0.149,   0.000,   0.000,   0.371,   0.000,   0.000,
                    2.043,   1.449,   8.817,   0.668,   1.969,   0.037,   0.000,
                    0.000,   4.457,   2.229,  14.856,  40.475,   0.000,   1.401,
                    0.000,   0.266,   0.000,   7.057,   0.223,   0.000,   0.557,
                    0.111,   0.000,   0.000,   0.059,   0.062,   0.000,   0.706,
                    0.037,   0.223,   0.223,   0.000,   0.000,   1.324,   0.683,
                    0.498,   0.655,   2.723,   0.903,   0.260,   0.000,   1.226,
                    0.557,  -0.100,   0.149,   0.958,   3.063,   1.128,   1.716,
                    0.978,   0.665,   0.000,   0.000,   2.266,   0.000,   0.000,
                    0.000,   1.894,   2.414,   0.557,   0.780,   0.000,   0.000,
                    0.404,   0.706,   0.000,   0.939,   1.218,   2.303,   7.190,
                    21.275,   8.769,   0.000,   0.520,   1.263,   0.000,   0.000,
                    0.000,   0.000,   0.446,   0.446,   0.122,   0.000,   0.264,
                    0.000,   0.000,   0.000,   0.443,   0.000,   1.082,   0.917,
                    2.377,   0.000,   1.411,   0.000,   2.524,   0.000,   0.059,
                    0.059,   2.043,   0.000,   1.037,   0.591,   1.917,   0.290,
                    2.414,   0.000,   1.114,   0.059,   3.146,  42.000,   0.000,
                    -2.561,  -2.679,   0.000,   1.262,   0.443,   0.441,   1.040,
                    2.971,   0.646,   1.187,   1.187,   6.132,   0.949,   0.176,
                    0.074,   0.000,   1.660,   0.865,   1.256,   2.077,   0.000,
                    0.000,   0.414,   0.511,   1.313,   0.874,   0.000,   0.059,
                    1.189,   0.000,   0.000,   0.000,   0.000,   0.000,   1.486,
                    1.486,   0.371,   0.000,   0.000,   0.000,   0.313,   0.296,
                    0.579,   0.000,   0.692,   0.065,   1.270,   0.889,   2.300,
                    0.000,   0.118,   1.411,   0.000,   6.967,   6.967,   0.989,
                    -0.083,   0.000,   0.475,   1.505,   0.903,   0.000,  17.589,
                    16.707,   0.000,   1.054,   0.680,   1.284,   3.289,   0.000,
                    0.000,   3.043,   4.346,   0.000,  32.143,   0.496,   1.052,
                    0.344,   0.363,   3.568,   3.269,   0.000,   2.414,   0.000,
                    0.000,  14.288,   0.000,   1.685,   1.021,   0.607,   1.475,
                    0.059,   2.266,   0.000,   0.300,   0.440,  -1.412,  -2.896,
                    0.000,   0.000,   1.093,   0.363,  -3.403,  -0.004,   1.505,
                    1.876,   0.748,   0.000,   2.992,   2.175,   0.669,   8.023,
                    0.000,   3.194,   6.760,   0.634,   0.294,   0.000,   0.000,
                    1.337,   0.000,   4.030,   2.396,   0.000,   0.000,   0.000,
                    0.000,   0.111,   0.111,   3.083,   0.111,   0.383,   0.469,
                    0.099,   0.103,   0.000,   0.000,   0.687,   2.957,   0.000,
                    0.264,   0.828,   0.000,   0.351,   0.000,  12.143,   1.476,
                    0.000,   2.406,   2.890,   0.505,   4.598,   0.000,   0.000,
                    1.119,   2.420,   0.000,   0.000,   0.044,   0.258,   0.000,
                    0.062,   1.077,   0.334,   1.465,   0.950,   0.071,  -1.609,
                    1.887,   1.961,   2.274,   0.000,   4.080,  -1.439,   1.805,
                    -2.662,   0.059,   0.000,   0.495,   1.281,   0.000,   0.000,
                    0.071,   0.071,   0.000,   2.637,   0.838,   2.769,   0.000,
                    0.000,   0.220,   0.334,   1.186,   0.631,   0.420,   0.002,
                    0.740,   0.843,   1.329,   0.000,   2.346,   0.059,   0.000,
                    0.397,   0.160,   1.597,   3.251,   3.417,   2.897,   0.000,
                    0.071,   0.130,   0.501,   2.252,   0.000,   0.839,   1.805,
                    0.000,   1.114,   0.817,   0.218,   0.479,   1.149,   0.592,
                    2.044,   0.000,   0.000,   0.000,  -0.765,   0.432,   0.000,
                    -0.173,   1.557,   0.814,   0.000,   0.000,   0.000,   3.157,
                    1.072,   0.181,   0.798,   0.000,   0.000,   0.074,   0.000,
                    0.000,  -1.372,   0.826,   4.716,   1.411,   0.000,   1.085,
                    1.582,   2.443,   9.657,   6.685,   1.838,   0.000,   0.588,
                    0.037,   0.436,   2.206,   0.000,   0.000,  -0.870,   0.000,
                    0.420,   0.662,  -1.023,   1.113,   1.783,   0.000,   0.191,
                    0.189,   2.112,   1.597,   0.000,   0.000,   1.830,   0.053,
                    0.000,   0.000,  -2.292,   0.452,   3.205,   0.000,   0.059,
                    2.407,   2.526,   0.149,   0.371,   1.040,   1.151,   0.334,
                    0.669,   2.749,   0.866,   0.000,   1.783,   0.771,   0.756,
                    2.708,   0.000,   0.362,   2.004,   0.611,   0.297,   0.000,
                    0.706,   0.000,   1.345,   0.000,   0.814,   3.231,   1.448,
                    2.017,   1.900,   0.192,   1.214,   0.000,   0.810,   0.000,
                    1.783,   0.000,   0.024,   1.125,  -0.932,   0.000,   0.334,
                    0.316,   1.203,   0.000,   0.000,   0.000,   0.852,   1.259,
                    5.523,   0.000,   1.313,   0.588,   0.000,   0.000,   0.000,
                    0.483,   0.000,  -2.575,  -2.072,   1.093,   1.336,   0.000,
                    0.000,   3.808,   2.035,   0.223,   0.000,   6.644,   0.000,
                    0.000,   0.000,   0.000,   0.446,   0.594,   0.297,   0.000,
                    0.000,   0.780,   0.780,   1.969,   1.693,   0.000,   1.486,
                    1.486,   1.552,   1.092,   0.373,   1.263,   1.189,   0.000,
                    0.000,   0.441,   0.037,   0.000,   1.894,  21.006,  19.000,
                    15.817,   1.449,   0.581,   0.000,   0.504,   1.371,   1.000,
                    0.000,   0.000,   1.486,  11.972,  11.049,  -0.306,   0.163,
                    0.218,   0.000,   0.000,   0.000,  -0.200,   1.040,   1.770,
                    2.903,   0.594,   0.000,   0.483,   1.032,   0.000,   0.000,
                    2.006,   0.966,   0.000,   1.365,  -0.131,   0.594,   0.000,
                    1.802,   1.738,   0.000,   0.000,   2.688,   0.374,   0.059,
                    0.370,   0.000,   1.411,   0.024,   0.000,   0.599,   1.427,
                    0.000,   0.000,   2.293,   2.628,   2.117,   1.151,   0.037,
                    0.420,   0.557,   3.380,   0.371,   0.000,   0.880,   2.448,
                    0.227,   2.081,   3.223,   0.964,   1.671,   2.229,   1.486,
                    0.046,   0.042,   0.836,   0.093,   2.180,   0.368,   1.434,
                    1.346,   0.000,   3.230,   1.001,   0.269,   1.857,   0.000,
                    0.564,   0.678,   1.102,   0.000,   1.552,   0.036,   0.443,
                    0.000,   0.064,   0.479,   1.231,   0.059,   2.303,   0.000,
                    1.003,   0.490,   2.756,   2.537,   8.172,   0.000,   1.411,
                    0.000,   1.631,   0.000,   0.985,   1.151,   0.000,   0.000,
                    1.086,   0.000,   0.210,   0.000,   0.000,   0.000,   3.343,
                    8.162,  10.008,   1.920,   2.793,   2.888,   1.149,   1.483,
                    0.000,   0.929,   0.334,   1.189,   1.478,   0.000,   3.170,
                    0.009,   0.000,   2.860,   3.677,   2.288,   2.024,   2.154,
                    0.000,   0.000,   0.000,   0.000,   0.000,   4.144,   1.812,
                    2.077,   0.000,   2.277,   3.657,   1.171,   1.158,   1.313,
                    2.934,   0.000,   0.059,   4.841,   1.403,   1.975,   0.000,
                    0.446,  29.000,   0.000,  29.000,   0.000,   0.908,   2.279,
                    3.127,   4.950,   0.000,   0.000,   0.111,   0.000,   1.632,
                    1.782,   2.036,   0.793,   0.758,   1.969,   0.712,   0.000,
                    3.065,   0.000, -31.369,   0.186,   0.523,   1.960,   2.523,
                    1.449,   0.024,   0.000,   0.149,   0.371,   0.000,   0.000,
                    0.000,   0.000,   6.649,   0.000,   1.607,   1.357,   0.000,
                    0.000,   0.520,   0.784,   1.893,   0.542,   3.566,   4.234,
                    1.579,   3.179,   1.747,   0.000,   0.000,   0.973,   0.743,
                    3.714,   0.000,   0.321,   0.536,   0.111,   0.966,   0.000,
                    0.891,   0.000,   1.189,   0.520,   1.574,   0.000,   0.365,
                    0.059,   1.229,   8.593,   0.000,   0.000,   0.530,   0.534,
                    0.000,   0.520,   0.711,   1.099,   2.115,   0.220,   0.000,
                    0.780,   0.984,   1.385,   1.227,   0.000,   0.000,   0.000,
                    3.639,   2.470,   5.000,   5.900,   6.210,   5.500,   2.239,
                    2.249,  -0.734,   0.000,   0.974,   0.000,   0.044,   4.136,
                    2.223,  -0.987,   0.000,   0.581,  -0.241,   0.644,   1.279,
                    2.343,   0.527,   2.652,   0.335,   0.425,   1.477,   0.150,
                    -1.132,   0.806,  -0.023,   0.006,   0.121,   0.000,   0.700,
                    -1.258,   0.894,   0.000,   3.000,   0.000,   0.107,  -0.092,
                    -0.032,   2.199,  -0.116,   0.607,   0.645,  -2.100,   0.000,
                    0.000,   0.000,   0.000,   0.088,   0.248,   1.781,   0.650,
                    -1.155,   0.576,   0.984,   1.755,   0.356,   1.194,   0.057,
                    -0.154,   2.312,   7.900,   0.000,   0.076,   0.070,  -0.514,
                    -0.020,   0.162,   0.000,   1.307,   0.979,   4.081,   8.000,
                    1.405,   0.000,   0.900,   1.200,   1.668,   1.200,   0.000,
                    1.412,   2.511,   1.832,  -0.900,   8.000,   0.000,   0.161,
                    0.585,   1.404,   1.068,  -1.154,   1.166,   3.390,   0.271,
                    0.000,   1.230,   1.177,   6.029,  -1.342,   2.020,   0.670,
                    0.977,   0.139,   0.000,   0.172,   0.449,   0.220,  12.339,
                    17.000,   7.935,   0.140,  55.700,   0.000,   0.251,   0.770,
                    3.382,   0.300,   0.301,   0.511,   1.536,   1.443,   0.756,
                    1.954,   8.953,   2.878,   0.241,   1.226,   0.000,   3.321,
                    1.463,   0.137,  -0.758,   2.599,   3.000,   3.342,   1.291,
                    0.000,   4.330,   3.700,   4.714,  -0.077,   1.002,   3.447,
                    0.549,   0.600,  -0.362,   0.000,   0.000,   0.000,   5.834,
                    9.158,   2.450,   0.905,   2.054,   1.245,   1.355,   1.282,
                    3.765,   0.000,   3.412,   0.000,   0.000,   0.000,   0.000,
                    0.555,   1.371,   8.223,   9.028,   3.835,   2.000,   0.002,
                    7.133,   1.072,   0.000,   0.810,   0.000,   0.000,   7.101,
                    16.992,   0.081,   1.841,   0.000,   5.131,   1.164,   4.356,
                    2.547,   2.015,   1.941,  10.185,   0.279,   3.596,   2.163,
                    1.925,   2.206,   0.095,   2.542,   3.353,   0.000,  -0.100,
                    5.227,   2.847,   0.164,   1.805,   1.226,   0.000,   0.849,
                    0.268,   2.771,   0.000,  -0.192,   1.222,   0.145,   4.197,
                    -0.011,   0.000,  -0.688,   0.000,   0.408,   3.988,   0.000,
                    0.254,   0.755,   0.289,   0.000,   0.000,   0.000,   0.362,
                    -0.200,   0.752,  -2.509,   0.126,   0.167,  -0.143,   0.858,
                    74.351,   1.005,   0.361,   0.158,  -0.873,   0.658,   0.166,
                    0.672,   0.537,   0.207,   1.586,   0.047,   0.046,   1.792,
                    0.998,   3.020,   0.829,   0.435,  -1.028,   2.426,   0.100,
                    3.378,   2.376,   2.364,   1.400,  -0.339,   3.990,  -0.268,
                    -0.568,  -1.337,  -4.388,   3.558,  -0.156,   0.000,   0.762,
                    -0.456,   0.562,   1.294,  -0.878,   3.489,   2.804,  -0.948,
                    0.912,   2.216,  -2.715,   0.000,   0.000,   2.715,   0.000,
                    0.000,   2.800,  12.500,  -2.724,   0.138,   0.060,   0.162,
                    2.378,   0.544,   0.371,  -1.162,   1.252,   0.519,   1.015,
                    0.889,   1.880,   0.406,   0.000,   0.602,   2.053,   0.000,
                    0.000,   5.143,   4.557,   0.000,   0.000,  -3.533,   0.230,
                    -1.560,  -0.400,   0.318,   1.671,   0.534,   0.663,   0.000,
                    0.164,  -0.600,   4.085,   0.000,   0.000,   1.700,   4.100,
                    1.406,   1.490,   1.039,   0.569,   2.548,   0.600,   1.300,
                    -0.487,   0.000,   2.431,   2.480,   1.200,   0.149,   2.650,
                    1.052,   0.144,   0.340,   2.716,   1.924,   2.032,   0.455,
                    2.082,   2.345,   0.000,   0.201,  14.940,  -0.910,   0.797,
                    -1.215,  -0.464,   0.213,   0.552,   0.123,   0.000,   0.186,
                    0.000,   0.000,   2.863,  -3.170,  -3.000,  -0.002,   1.071,
                    -0.084,   1.747,   0.639,   1.870,   0.945,   0.318,   2.200,
                    -1.114,   2.127,   2.600,   0.121,   3.863,  -2.337,   0.461,
                    -0.989,   0.000,   2.037,   0.000,   1.521,   0.067,   0.500,
                    3.742,   0.000,  -0.190,   2.278,   0.000,   0.615,   0.202,
                    1.532,   0.867,   0.278,   0.000,  -0.303,   1.311,   0.300,
                    0.016,   2.594,   5.075,   0.241,   8.211,   0.000,   0.104,
                    0.007,   0.610,   1.017,   7.195,  32.200,   4.200,   4.492,
                    1.053,   0.237,   1.075,   1.282,   2.749,   0.000,   1.453,
                    0.051,   1.488,   2.108,   0.400,   0.187,   0.000,   0.000,
                    2.253,   2.062,   1.065,   0.000,   0.000,   2.141,   0.365,
                    0.360,   1.334,   1.720,   1.012,   1.334,  -0.844,   1.072,
                    1.722,   1.476,   1.058,  -0.347,   0.000,   0.779,   1.264,
                    0.698,   3.596,  -1.341,   2.388,   1.768,   0.104,   0.520,
                    -2.200,   2.400,   0.720,   4.340,   0.432,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.000,   0.000,   0.000,   0.000,   0.000,   0.000,   0.000,
                    0.682,   0.583,  -0.788,   2.419,   7.875,   1.249,   0.000,
                    1.109,   0.000,   1.097,   0.377,   6.806,   4.844,  -0.622,
                    -1.500,   0.272,   0.887,   1.000,   0.014,   3.359,   2.220,
                    0.164,  -0.500,   0.136,   1.247,   0.000,   0.756,   0.000,
                    0.000,   0.000,   1.375,   0.200,   2.300,   1.300,   2.000,
                    3.500,   1.900,   2.000,   0.000,   1.600,  13.300,   3.000,
                    4.000,   0.300,   2.000,   1.300,   1.200,   6.000,   2.500,
                    0.000,   0.000,   1.000,   1.500,   2.500,   2.400,   1.400,
                    3.900,   3.600,   2.000,   1.000,   2.500,   0.900,   2.000,
                    1.500,   0.000,   0.400,   0.500,   0.100,   2.000,   0.000,
                    0.200,   6.000,   2.000,   0.000,   0.000,   3.200,   3.000,
                    2.100,   1.700,   3.500,   0.000,   0.400,   0.000,   0.000,
                    1.500,   1.100,   0.000,   0.000,   2.000,   0.900,   0.800,
                    0.800,   1.000,   3.300,   1.200,   0.500,   2.600,   4.200,
                    0.000,   0.100,   0.500,   0.100,   1.500,   0.000,   0.000,
                    0.000,   0.800,   1.500,   3.000,   1.700,   4.500,   2.000,
                    2.000,  20.000,   2.300,   0.500,   1.000,   0.500,   0.700,
                    4.700,   0.000,   3.400,   2.000,   1.000,   1.700,   2.200,
                    2.500,   2.500,   2.500,   2.000,   0.000,   2.500,   1.300,
                    1.000,   2.500,   1.500,   1.200,   2.600,  73.400,   1.300,
                    1.800,   2.500,   0.700,   0.800,   0.000,   3.700,   2.700,
                    1.000,   1.000,   0.000,   3.900,   1.200,   1.500,   1.100,
                    4.600,  17.500,   2.100,   1.000,   1.600,   4.000,   2.000,
                    0.100,   2.500,   2.000,   1.500,   4.000,   3.000,   3.500,
                    0.800,   3.100,   1.300,   0.700,   1.200,   0.400,   0.200,
                    1.300,   0.000,   0.000,   2.500,   1.200,   1.500,   2.900,
                    3.000,   1.200,   2.000,   3.000,   0.400,   1.000,   4.000,
                    1.700,   3.000,   5.000,   2.000,   1.800,   2.000,   0.100,
                    0.500,   3.600,   2.400,   0.000,   0.600,   0.800,   1.800,
                    0.800,   1.500,   0.500,   2.800,   0.200,   1.900,   0.400,
                    0.500,   0.100,   4.200,   2.000,   1.800,   3.500,   6.500,
                    0.000,   1.700,   1.400,   6.100,   1.000,   0.000,   4.200,
                    3.000,   1.300,   5.000,   1.400,   3.000,   0.600,   7.000,
                    2.500,   6.000,   1.500,   0.000,   2.500,   2.600,   1.000,
                    1.500,   0.000,   4.000,   3.100,   1.000,   0.500,   0.000,
                    0.000,   0.500,   0.500,   0.700,   3.600,   0.000,   2.800,
                    1.000,   0.000,   4.000,   1.300,   2.000,   2.000,   1.100,
                    33.000,   1.000,   1.200,   5.000,   2.000,   0.000,   2.400,
                    1.900,   2.700,   1.400,   1.200,   2.500,   2.300,   3.000,
                    1.100,   2.000,   1.500,   3.700,   5.100,   1.500,   1.200,
                    3.100,   2.600,   6.200,   2.400,   2.400,   3.100,   0.000,
                    3.100,   0.800,   0.000,   0.000,   0.000,  11.000,   1.000,
                    1.100,   1.200,  18.800,   4.300,   0.000,   4.400,   4.000,
                    0.000,   0.000,   4.000,   0.000,   0.000,   0.500,  31.638,
                    10.504, -75.565, -31.538,   0.000,   0.000, 188.523,   0.000,
                    6.000,   0.000,  12.000,   6.000,  10.000,  26.000])/Sbase

    
    Gsh = np.zeros(n_bus)
    
    Bsh = np.zeros(n_bus)
    Bsh[197] = -117/Sbase
    
    R_br = np.array([0.000640,   0.000360,   0.000360,   0.000340,   0.000340,
                    0.000810,   0.000750,   0.000750,   0.000500,   0.000640,
                    0.000232,   0.000232,   0.000350,   0.000590,   0.000400,
                    0.000310,   0.000310,   0.014341,   0.012190,   0.011781,
                    0.001525,   0.001651,   0.001651,   0.001880,   0.001880,
                    0.002769,   0.013450,   0.009091,   0.008680,   0.016961,
                    0.005120,   0.010680,   0.014651,   0.014880,   0.011159,
                    0.005517,   0.011760,   0.016818,   0.013843,   0.016760,
                    0.012690,   0.014091,   0.003719,   0.001940,   0.001500,
                    0.002940,   0.001690,   0.001370,   0.002310,   0.002070,
                    0.001500,   0.004690,   0.003310,   0.002190,   0.016740,
                    0.005700,   0.004419,   0.014591,   0.012110,   0.009773,
                    0.010680,   0.005700,   0.005700,   0.005250,   0.000021,
                    0.000930,   0.002050,   0.007831,   0.002930,   0.001550,
                    0.001550,   0.003841,   0.002980,   0.003580,   0.003360,
                    0.002990,   0.001120,   0.000750,   0.010479,   0.008820,
                    0.011219,   0.011630,   0.005450,   0.005450,   0.008370,
                    0.003159,   0.006979,   0.003740,   0.007690,   0.005560,
                    0.004521,   0.001510,   0.005740,   0.007070,   0.012130,
                    0.002479,   0.002401,   0.001670,   0.001090,   0.002259,
                    0.000940,   0.001130,   0.001760,   0.004876,   0.010930,
                    0.001942,   0.004318,   0.000661,   0.005372,   0.000826,
                    0.015888,   0.002169,   0.017727,   0.005930,   0.003905,
                    0.008161,   0.008161,   0.005186,   0.002128,   0.004628,
                    0.011219,   0.006012,   0.002066,   0.001364,   0.000579,
                    0.000847,   0.000847,   0.001591,   0.003141,   0.004545,
                    0.001446,   0.003822,   0.007769,   0.004463,   0.005888,
                    0.003161,   0.001839,   0.001715,   0.003409,   0.001839,
                    0.005702,   0.004587,   0.002149,   0.000785,   0.000785,
                    0.002413,   0.002131,   0.000913,   0.003119,   0.000919,
                    0.015450,   0.015450,   0.014570,   0.014570,   0.006031,
                    0.011031,   0.005831,   0.018490,   0.010581,   0.012521,
                    0.005060,   0.010430,   0.007702,   0.002256,   0.009147,
                    0.002440,   0.003550,   0.013120,   0.009341,   0.006141,
                    0.004380,   0.000479,   0.000479,   0.010539,   0.004690,
                    0.006469,   0.001050,   0.006099,   0.002211,   0.005169,
                    0.002401,   0.002740,   0.001700,   0.001360,   0.003710,
                    0.018820,   0.027560,   0.019050,   0.007950,   0.001800,
                    0.007310,   0.005640,   0.005700,   0.001310,   0.002240,
                    0.002100,   0.001500,   0.001330,   0.001330,   0.001570,
                    0.001570,   0.001008,   0.000790,   0.001651,   0.001651,
                    0.001651,   0.001651,   0.001651,   0.001651,   0.001500,
                    0.001651,   0.001651,   0.001651,   0.000870,   0.000870,
                    0.001240,   0.001240,   0.001651,   0.001651,   0.001550,
                    0.001570,   0.001651,   0.001651,   0.001570,   0.001651,
                    0.001500,   0.001651,   0.001370,   0.001370,   0.001651,
                    0.001651,   0.001651,   0.001401,   0.001401,   0.001510,
                    0.001570,   0.001490,   0.001570,   0.001630,   0.001350,
                    0.001350,   0.001530,   0.001570,   0.001400,   0.001390,
                    0.001450,   0.001651,   0.001570,   0.001630,   0.001430,
                    0.001390,   0.001400,   0.001630,   0.001610,   0.002070,
                    0.001651,   0.001450,   0.001560,   0.001560,   0.001651,
                    0.001450,   0.001450,   0.001450,   0.001450,   0.001651,
                    0.001450,   0.001651,   0.001450,   0.001651,   0.001651,
                    0.001450,   0.001651,   0.001450,   0.001651,   0.002070,
                    0.001860,   0.001450,   0.001450,   0.001450,   0.001450,
                    0.001560,   0.001560,   0.001450,   0.001450,   0.001370,
                    0.001370,   0.001560,   0.001651,   0.001450,   0.001651,
                    0.001450,   0.001450,   0.001651,   0.001450,   0.001651,
                    0.001510,   0.001530,   0.001520,   0.001510,   0.001510,
                    0.001610,   0.001610,   0.001450,   0.001450,   0.001651,
                    0.003409,   0.011240,   0.005620,   0.001630,   0.001651,
                    0.001651,   0.001651,   0.001651,   0.001651,   0.000289,
                    0.001651,   0.001010,   0.001010,   0.001380,   0.001630,
                    0.001651,   0.001651,   0.002810,   0.002810,   0.001651,
                    0.001010,   0.001530,   0.001651,   0.001651,   0.001651,
                    0.001651,   0.001651,   0.001651,   0.001651,   0.001651,
                    0.001550,   0.001370,   0.001290,   0.001290,   0.003700,
                    0.001591,   0.001630,   0.001630,   0.000590,   0.001591,
                    0.001591,   0.000890,   0.001610,   0.001610,   0.000590,
                    0.002771,   0.001570,   0.001610,   0.001651,   0.001370,
                    0.001370,   0.001550,   0.001610,   0.054463,   0.023719,
                    0.024711,   0.042066,   0.061818,   0.059256,   0.033058,
                    0.021488,   0.047273,   0.074628,   0.052231,   0.045446,
                    0.037190,   0.041322,   0.023967,   0.024380,   0.035537,
                    0.070248,   0.021488,   0.068256,   0.034628,   0.029174,
                    0.037190,   0.082066,   0.044628,   0.041322,   0.039256,
                    0.056281,   0.038843,   0.023058,   0.046777,   0.040496,
                    0.057016,   0.016364,   0.056612,   0.009917,   0.053331,
                    0.013967,   0.059917,   0.074380,   0.038843,   0.037017,
                    0.017190,   0.014050,   0.021488,   0.019008,   0.047934,
                    0.042314,   0.037273,   0.076859,   0.056529,   0.028256,
                    0.043140,   0.019008,   0.030744,   0.021397,   0.012397,
                    0.012231,   0.019587,   0.006860,   0.005537,   0.002727,
                    0.023554,   0.023388,   0.015446,   0.019669,   0.031818,
                    0.032562,   0.002479,   0.040579,   0.013967,   0.043058,
                    0.027364,   0.024554,   0.055537,   0.111818,   0.030826,
                    0.019669,   0.020909,   0.023636,   0.002479,   0.043140,
                    0.022810,   0.029339,   0.013719,   0.029752,   0.031488,
                    0.047364,   0.024298,   0.006446,   0.037017,   0.022562,
                    0.059256,   0.027438,   0.048934,   0.023802,   0.007107,
                    0.013719,   0.020909,   0.011157,   0.006686,   0.004554,
                    0.006281,   0.006198,   0.004380,   0.007273,   0.007273,
                    0.005620,   0.007438,   0.004298,   0.013636,   0.010826,
                    0.003876,   0.010165,   0.001570,   0.002066,   0.017934,
                    0.014050,   0.004380,   0.004380,   0.007273,   0.001240,
                    0.002066,   0.014050,   0.004876,   0.013140,   0.008678,
                    0.002066,   0.008099,   0.012479,   0.000413,   0.002521,
                    0.001372,   0.012397,   0.004380,   0.012727,   0.000504,
                    0.004207,   0.001397,   0.005289,   0.003967,   0.000413,
                    0.003223,   0.003141,   0.000579,   0.004380,   0.010174,
                    0.006612,   0.005793,   0.002314,   0.012727,   0.003554,
                    0.000504,   0.002149,   0.002397,   0.003554,   0.009917,
                    0.013388,   0.019669,   0.002636,   0.008512,   0.004628,
                    0.018934,   0.016198,   0.008678,   0.020661,   0.000826,
                    0.004132,   0.000570,   0.000570,   0.004132,   0.003967,
                    0.033223,   0.021240,   0.068603,   0.041744,   0.013719,
                    0.017769,   0.032397,   0.015041,   0.011818,   0.009174,
                    0.007851,   0.006364,   0.022397,   0.078678,   0.005950,
                    0.047190,   0.004959,   0.059256,   0.002893,   0.003967,
                    0.017017,   0.011066,   0.009421,   0.060909,   0.006777,
                    0.006529,   0.039669,   0.023141,   0.053223,   0.039421,
                    0.020174,   0.035289,   0.003314,   0.003471,   0.039091,
                    0.006529,   0.032893,   0.011653,   0.009008,   0.023141,
                    0.011744,   0.006942,   0.003554,   0.006364,   0.006942,
                    0.019752,   0.031157,   0.005372,   0.016777,   0.026860,
                    0.027273,   0.013719,   0.005446,   0.006198,   0.010744,
                    0.005702,   0.002231,   0.003058,   0.005793,   0.018017,
                    0.004628,   0.007107,   0.001066,   0.009917,   0.004628,
                    0.003719,   0.000579,   0.003058,   0.005289,   0.007364,
                    0.002149,   0.005620,   0.004876,   0.005950,   0.004050,
                    0.007521,   0.013876,   0.016612,   0.005041,   0.010578,
                    0.002636,   0.000083,   0.003388,   0.002231,   0.010661,
                    0.015793,   0.005702,   0.001322,   0.018760,   0.017107,
                    0.003388,   0.001322,   0.002562,   0.007686,   0.002562,
                    0.004793,   0.007851,   0.009091,   0.012810,   0.006124,
                    0.011983,   0.005620,   0.023802,   0.003876,   0.004959,
                    0.005041,   0.030000,   0.025207,   0.001488,   0.008430,
                    0.019504,   0.014876,   0.022636,   0.014132,   0.004405,
                    0.016529,   0.017851,   0.030331,   0.007190,   0.017107,
                    0.031240,   0.010000,   0.028017,   0.004132,   0.025950,
                    0.040413,   0.002562,   0.014132,   0.007107,   0.005793,
                    0.010496,   0.054554,   0.060579,   0.002727,   0.011983,
                    0.010744,   0.009826,   0.012984,   0.003141,   0.003554,
                    0.013719,   0.001636,   0.019669,   0.003554,   0.028678,
                    0.006860,   0.022314,   0.014711,   0.019504,   0.020579,
                    0.008512,   0.004876,   0.008430,   0.035289,   0.024298,
                    0.001818,   0.001744,   0.007884,   0.001074,   0.002562,
                    0.000008,   0.019504,   0.035702,   0.044297,   0.018934,
                    0.010992,   0.012149,   0.029091,   0.033876,   0.019826,
                    0.024215,   0.022231,   0.008347,   0.007603,   0.011818,
                    0.016198,   0.019917,   0.010661,   0.015868,   0.010744,
                    0.027016,   0.019091,   0.019752,   0.025041,   0.007603,
                    0.023141,   0.038843,   0.018347,   0.030331,   0.009752,
                    0.036777,   0.021397,   0.001983,   0.005620,   0.003058,
                    0.009587,   0.010661,   0.004298,   0.091653,   0.031240,
                    0.004793,   0.009174,   0.006777,   0.016612,   0.011322,
                    0.012893,   0.043967,   0.024554,   0.021653,   0.044050,
                    0.018256,   0.020496,   0.033388,   0.002727,   0.006942,
                    0.059917,   0.100413,   0.051984,   0.028934,   0.030331,
                    0.006612,   0.026942,   0.020000,   0.027016,   0.049917,
                    0.007851,   0.040174,   0.003967,   0.022066,   0.037364,
                    0.002636,   0.039504,   0.035372,   0.019752,   0.031653,
                    0.040661,   0.041397,   0.015703,   0.028017,   0.026777,
                    0.038603,   0.019091,   0.067686,   0.061066,   0.006124,
                    0.050992,   0.031240,   0.029174,   0.021653,   0.018678,
                    0.026777,   0.004298,   0.039421,   0.008182,   0.043388,
                    0.018256,   0.028678,   0.025446,   0.018182,   0.027934,
                    0.023636,   0.002231,   0.000983,   0.016388,   0.000174,
                    0.015454,   0.000008,   0.036281,   0.030413,   0.026777,
                    0.022562,   0.086777,   0.072066,   0.050000,   0.011240,
                    0.016364,   0.041157,   0.023058,   0.023058,   0.020174,
                    0.048934,   0.031983,   0.039174,   0.038512,   0.061397,
                    0.003719,   0.048430,   0.012066,   0.037603,   0.010752,
                    0.012703,   0.034851,   0.014207,   0.005620,   0.004463,
                    0.002562,   0.013058,   0.011744,   0.025620,   0.029339,
                    0.011322,   0.002397,   0.002983,   0.018843,   0.025124,
                    0.004132,   0.023141,   0.007769,   0.003554,   0.010826,
                    0.004554,   0.001240,   0.003141,   0.033223,   0.013636,
                    0.013388,   0.004628,   0.012636,   0.019669,   0.028256,
                    0.021818,   0.058347,   0.005702,   0.033223,   0.063719,
                    0.013719,   0.003802,   0.002314,   0.004132,   0.003967,
                    0.032314,   0.040413,   0.003967,   0.013058,   0.030331,
                    0.037934,   0.009587,   0.005446,   0.026364,   0.019174,
                    0.030661,   0.031066,   0.019174,   0.029174,   0.005207,
                    0.011066,   0.020331,   0.004959,   0.001066,   0.031157,
                    0.005124,   0.032149,   0.019008,   0.045793,   0.024628,
                    0.058182,   0.010174,   0.003471,   0.006777,   0.014959,
                    0.063471,   0.044554,   0.069091,   0.014793,   0.018430,
                    0.044628,   0.038430,   0.037686,   0.055703,   0.040992,
                    0.007934,   0.008512,   0.009421,   0.007851,   0.027769,
                    0.010826,   0.023777,   0.008347,   0.018430,   0.020661,
                    0.048760,   0.010496,   0.026612,   0.017521,   0.037521,
                    0.035537,   0.023802,   0.018017,   0.001653,   0.010826,
                    0.016777,   0.014554,   0.083058,   0.014132,   0.026380,
                    0.031240,   0.034876,   0.002066,   0.019917,   0.017190,
                    0.012066,   0.058760,   0.045207,   0.013058,   0.044554,
                    0.022066,   0.008760,   0.003967,   0.016198,   0.015950,
                    0.011240,   0.010826,   0.003719,   0.022636,   0.136198,
                    0.006124,   0.004628,   0.007438,   0.003223,   0.027107,
                    0.009669,   0.045446,   0.026281,   0.010909,   0.035207,
                    0.022397,   0.021322,   0.076942,   0.031818,   0.009174,
                    0.023554,   0.022231,   0.022066,   0.025372,   0.014050,
                    0.020744,   0.009669,   0.006198,   0.015793,   0.001744,
                    0.020579,   0.028017,   0.008678,   0.033058,   0.032983,
                    0.020661,   0.014793,   0.007438,   0.008256,   0.002562,
                    0.005289,   0.038934,   0.021570,   0.029669,   0.009826,
                    0.024380,   0.017769,   0.008182,   0.012397,   0.008512,
                    0.006364,   0.017934,   0.001488,   0.000579,   0.028678,
                    0.015703,   0.021322,   0.008182,   0.011488,   0.006198,
                    0.018099,   0.005702,   0.019174,   0.023058,   0.000157,
                    0.030248,   0.013587,   0.051901,   0.015207,   0.008512,
                    0.027016,   0.018512,   0.013719,   0.022893,   0.025207,
                    0.005793,   0.017769,   0.020579,   0.009008,   0.023967,
                    0.023223,   0.008843,   0.014628,   0.053314,   0.040661,
                    0.027769,   0.002893,   0.040000,   0.038256,   0.028934,
                    0.004132,   0.069091,   0.039917,   0.012810,   0.013223,
                    0.031322,   0.002479,   0.002479,   0.000331,   0.016281,
                    0.034711,   0.015124,   0.029496,   0.010248,   0.031240,
                    0.024463,   0.057364,   0.013636,   0.003058,   0.004463,
                    0.014050,   0.007603,   0.004463,   0.005702,   0.021984,
                    0.016198,   0.039174,   0.009587,   0.009669,   0.047769,
                    0.022727,   0.024628,   0.005702,   0.047190,   0.007769,
                    0.016859,   0.003554,   0.005868,   0.003802,   0.004050,
                    0.003554,   0.017769,   0.017769,   0.021744,   0.017190,
                    0.017190,   0.020744,   0.001570,   0.038256,   0.024876,
                    0.015207,   0.000413,   0.002810,   0.022893,   0.019339,
                    0.004554,   0.034876,   0.020992,   0.010826,   0.019174,
                    0.026281,   0.002975,   0.036116,   0.000000,   0.016198,
                    0.000165,   0.016529,   0.000496,   0.038678,   0.005372,
                    0.039091,   0.000083,   0.015950,   0.031736,   0.000496,
                    0.015950,   0.031736,   0.000496,   0.048182,   0.040165,
                    0.011653,   0.011653,   0.002727,   0.007603,   0.000331,
                    0.018182,   0.026033,   0.013884,   0.017355,   0.006860,
                    0.004298,   0.005785,   0.006033,   0.002314,   0.005041,
                    0.010248,   0.005124,   0.005041,   0.010248,   0.005124,
                    0.040165,   0.022479,   0.018760,   0.014628,   0.006281,
                    0.004380,   0.006281,   0.005041,   0.004380,   0.012479,
                    0.012479,   0.002231,   0.004132,   0.004876,   0.001240,
                    0.005537,   0.041570,   0.003967,   0.015868,   0.021570,
                    0.002397,   0.002397,   0.010496,   0.024793,   0.010826,
                    0.008347,   0.024793,   0.034876,   0.008182,   0.056033,
                    0.038265,   0.029256,   0.019752,   0.021818,   0.036281,
                    0.019917,   0.007769,   0.047603,   0.052479,   0.032645,
                    0.023719,   0.008264,   0.019917,   0.000992,   0.013554,
                    0.005372,   0.020248,   0.037273,   0.018760,   0.015868,
                    0.006777,   0.015372,   0.006860,   0.006860,   0.053058,
                    0.049174,   0.034298,   0.000496,   0.004132,   0.004132,
                    0.001157,   0.000083,   0.017355,   0.024546,   0.026612,
                    0.031653,   0.045455,   0.002975,   0.013471,   0.008926,
                    0.020413,   0.000083,   0.006529,   0.000165,   0.002562,
                    0.004050,   0.000744,   0.010826,   0.000413,   0.001240,
                    0.004545,   0.004545,   0.004132,   0.006777,   0.006777,
                    0.037025,   0.014876,   0.007603,   0.000992,   0.006198,
                    0.007273,   0.019008,   0.019091,   0.007025,   0.010826,
                    0.008595,   0.028347,   0.008595,   0.015703,   0.014711,
                    0.003058,   0.011901,   0.008843,   0.026777,   0.031983,
                    0.036116,   0.023636,   0.022810,   0.000000,   0.009504,
                    0.009504,   0.004298,   0.040579,   0.017521,   0.003141,
                    0.000000,   0.000000,   0.005537,   0.002149,   0.005455,
                    0.004959,   0.019008,   0.005041,   0.014215,   0.021570,
                    0.015454,   0.012645,   0.002727,   0.025372,   0.001488,
                    0.013223,   0.008182,   0.009008,   0.003306,   0.004711,
                    0.002975,   0.015868,   0.026116,   0.004132,   0.008595,
                    0.002810,   0.010331,   0.020579,   0.004628,   0.019174,
                    0.018017,   0.006694,   0.017025,   0.040909,   0.006612,
                    0.025124,   0.014380,   0.028760,   0.002645,   0.019587,
                    0.003719,   0.024215,   0.007686,   0.007686,   0.019256,
                    0.002975,   0.004050,   0.020331,   0.005124,   0.015620,
                    0.017355,   0.002893,   0.021984,   0.016364,   0.008843,
                    0.004793,   0.017934,   0.013471,   0.000083,   0.004628,
                    0.040248,   0.019669,   0.006942,   0.000661,   0.002727,
                    0.000165,   0.002149,   0.012562,   0.010000,   0.005868,
                    0.000083,   0.000331,   0.004463,   0.013967,   0.022810,
                    0.006694,   0.004545,   0.005702,   0.013636,   0.017851,
                    0.004876,   0.000909,   0.016694,   0.000826,   0.002149,
                    0.005950,   0.000000,   0.002397,   0.008926,   0.004959,
                    0.002314,   0.000826,   0.007934,   0.000248,   0.008017,
                    0.008099,   0.014463,   0.010083,   0.003719,   0.002231,
                    0.009587,   0.000992,   0.005785,   0.000826,   0.011901,
                    0.004215,   0.005207,   0.004793,   0.001901,   0.000083,
                    0.001405,   0.001488,   0.006777,   0.008099,   0.000496,
                    0.000909,   0.001736,   0.012975,   0.003802,   0.006694,
                    0.004463,   0.001157,   0.002314,   0.004463,   0.003306,
                    0.001405,   0.002397,   0.003802,   0.018347,   0.000992,
                    0.001488,   0.004298,   0.001322,   0.000000,   0.002727,
                    0.002231,   0.000083,   0.003141,   0.000000,   0.001074,
                    0.004545,   0.001157,   0.001074,   0.003884,   0.000661,
                    0.009339,   0.005702,   0.005537,   0.015041,   0.000083,
                    0.000165,   0.000496,   0.000496,   0.012479,   0.000909,
                    0.000413,   0.008760,   0.003802,   0.000826,   0.002066,
                    0.000413,   0.001488,   0.004959,   0.000661,   0.011983,
                    0.004545,   0.004215,   0.005702,   0.001322,   0.001322,
                    0.002893,   0.002893,   0.003719,   0.004628,   0.002645,
                    0.008347,   0.001570,   0.007438,   0.000992,   0.000744,
                    0.000248,   0.008760,   0.010083,   0.011901,   0.004380,
                    0.010083,   0.008099,   0.004132,   0.002397,   0.003141,
                    0.003058,   0.003141,   0.001074,   0.001818,   0.000579,
                    0.003141,   0.004736,   0.004050,   0.001983,   0.004959,
                    0.001074,   0.002314,   0.005702,   0.000083,   0.001901,
                    0.001405,   0.008099,   0.000496,   0.007521,   0.000331,
                    0.002397,   0.008099,   0.003636,   0.001570,   0.002314,
                    0.001983,   0.005620,   0.004959,   0.001074,   0.005207,
                    0.013388,   0.007603,   0.001074,   0.008760,   0.011157,
                    0.002314,   0.001058,   0.003223,   0.005455,   0.004215,
                    0.002066,   0.000661,   0.002893,   0.005950,   0.005289,
                    0.001405,   0.005124,   0.000331,   0.006529,   0.008017,
                    0.007273,   0.024050,   0.003141,   0.005785,   0.002066,
                    0.000248,   0.007438,   0.012066,   0.000496,   0.008099,
                    0.007686,   0.000579,   0.002231,   0.002727,   0.003223,
                    0.005289,   0.005289,   0.019256,   0.010413,   0.001653,
                    0.004711,   0.004793,   0.007438,   0.000083,   0.000744,
                    0.006777,   0.000909,   0.012397,   0.009421,   0.015124,
                    0.015289,   0.009339,   0.002149,   0.000000,   0.005041,
                    0.005785,   0.034132,   0.011074,   0.006198,   0.001240,
                    0.007686,   0.012479,   0.021901,   0.004876,   0.006198,
                    0.004050,   0.010248,   0.009587,   0.001405,   0.007769,
                    0.008182,   0.001488,   0.006281,   0.006198,   0.000826,
                    0.000744,   0.001983,   0.003471,   0.006446,   0.004793,
                    0.008843,   0.005041,   0.003554,   0.016033,   0.012645,
                    0.004298,   0.017273,   0.012810,   0.008926,   0.009008,
                    0.008512,   0.008512,   0.004298,   0.004298,   0.004959,
                    0.010992,   0.004298,   0.016033,   0.006198,   0.004545,
                    0.004132,   0.003471,   0.004628,   0.004711,   0.003554,
                    0.001570,   0.002479,   0.006116,   0.002066,   0.002397,
                    0.002397,   0.003719,   0.006281,   0.000083,   0.007603,
                    0.003141,   0.004545,   0.000496,   0.002562,   0.001653,
                    0.003223,   0.003058,   0.000000,   0.001405,   0.010744,
                    0.008264,   0.014711,   0.009917,   0.002231,   0.006033,
                    0.003058,   0.006529,   0.002397,   0.008678,   0.008347,
                    0.004711,   0.024050,   0.003471,   0.001488,   0.000909,
                    0.010331,   0.008430,   0.001488,   0.001983,   0.004380,
                    0.000661,   0.000165,   0.004628,   0.000826,   0.002562,
                    0.000992,   0.005289,   0.000826,   0.000826,   0.000826,
                    0.000744,   0.004959,   0.007190,   0.008017,   0.006033,
                    0.007686,   0.001405,   0.004876,   0.003719,   0.002727,
                    0.001157,   0.007686,   0.011240,   0.005868,   0.000165,
                    0.001983,   0.063554,   0.030413,   0.099008,   0.001983,
                    0.016116,   0.022893,   0.024793,   0.022727,   0.014711,
                    0.096446,   0.015124,   0.037273,   0.000579,   0.018017,
                    0.012397,   0.016529,   0.014545,   0.007190,   0.007190,
                    0.006612,   0.016364,   0.019504,   0.001322,   0.001322,
                    0.028430,   0.037769,   0.003306,   0.038182,   0.028430,
                    0.005207,   0.003306,   0.040496,   0.025785,   0.024463,
                    0.024463,   0.024793,   0.001653,   0.001074,   0.030744,
                    0.002314,   0.011240,   0.007769,   0.050000,   0.026033,
                    0.042397,   0.004380,   0.032479,   0.050248,   0.025372,
                    0.013306,   0.010000,   0.005868,   0.006281,   0.009421,
                    0.005207,   0.002149,   0.002149,   0.033058,   0.017769,
                    0.022479,   0.014545,   0.029917,   0.011157,   0.034628,
                    0.021074,   0.030413,   0.007603,   0.014711,   0.010248,
                    0.004628,   0.003388,   0.003306,   0.034380,   0.034380,
                    0.053471,   0.008678,   0.017025,   0.082314,   0.047190,
                    0.058099,   0.013636,   0.000826,   0.000826,   0.001157,
                    0.023719,   0.004132,   0.035950,   0.000248,   0.017851,
                    0.018182,   0.025124,   0.011570,   0.024380,   0.004132,
                    0.005702,   0.000331,   0.019174,   0.000826,   0.000909,
                    0.001818,   0.001901,   0.004628,   0.008264,   0.005372,
                    0.003471,   0.003636,   0.003636,   0.001074,   0.001157,
                    0.001818,   0.002479,   0.017934,   0.006281,   0.013223,
                    0.003719,   0.002066,   0.011983,   0.012314,   0.003884,
                    0.000661,   0.000909,   0.043223,   0.009256,   0.007686,
                    0.034793,   0.009091,   0.004132,   0.087769,   0.020000,
                    0.039339,   0.043223,   0.015703,   0.036777,   0.009339,
                    0.032645,   0.026529,   0.003884,   0.035785,   0.003967,
                    0.007107,   0.007025,   0.019091,   0.006529,   0.010000,
                    0.038678,   0.002645,   0.014050,   0.012727,   0.002149,
                    0.007686,   0.002645,   0.002645,   0.015124,   0.008926,
                    0.041240,   0.005207,   0.015207,   0.004711,   0.010909,
                    0.031901,   0.001653,   0.006529,   0.021984,   0.045124,
                    0.045703,   0.000826,   0.004298,   0.058099,   0.003884,
                    0.001653,   0.001653,   0.003967,   0.018017,   0.017851,
                    0.006777,   0.006612,   0.020000,   0.000579,   0.043719,
                    0.032562,   0.049422,   0.011735,   0.002645,   0.010578,
                    0.005207,   0.001570,   0.001488,   0.000496,   0.001074,
                    0.009917,   0.023141,   0.022149,   0.020827,   0.018512,
                    0.000496,   0.006942,   0.006860,   0.000496,   0.005702,
                    0.021322,   0.002562,   0.001983,   0.000579,   0.005124,
                    0.003636,   0.005950,   0.010000,   0.023388,   0.006612,
                    0.003636,   0.008017,   0.012645,   0.005868,   0.027355,
                    0.001736,   0.021570,   0.003141,   0.011901,   0.006777,
                    0.025289,   0.003141,   0.034298,   0.014545,   0.019587,
                    0.003554,   0.001983,   0.000909,   0.004298,   0.003141,
                    0.009587,   0.010413,   0.002645,   0.010083,   0.012314,
                    0.001901,   0.003554,   0.001322,   0.003471,   0.003388,
                    0.009421,   0.007107,   0.000992,   0.001240,   0.000826,
                    0.001818,   0.003223,   0.003719,   0.002397,   0.012810,
                    0.011983,   0.004959,   0.001074,   0.003388,   0.000165,
                    0.005289,   0.004215,   0.010992,   0.005785,   0.000248,
                    0.000413,   0.002727,   0.004380,   0.000331,   0.001240,
                    0.003388,   0.002231,   0.005289,   0.002397,   0.000992,
                    0.004959,   0.002314,   0.008182,   0.004711,   0.000909,
                    0.006529,   0.012314,   0.025868,   0.004959,   0.001405,
                    0.004132,   0.017603,   0.006364,   0.008595,   0.026198,
                    0.006364,   0.019752,   0.001901,   0.002645,   0.000083,
                    0.011735,   0.005620,   0.001901,   0.003802,   0.002645,
                    0.000083,   0.001240,   0.002397,   0.004132,   0.007107,
                    0.000744,   0.000909,   0.003141,   0.000413,   0.000413,
                    0.000661,   0.012893,   0.019422,   0.020827,   0.001736,
                    0.007603,   0.001736,   0.004132,   0.012479,   0.003471,
                    0.003636,   0.000000,   0.004711,   0.004463,   0.027686,
                    0.000826,   0.000826,   0.005537,   0.006116,   0.003141,
                    0.000744,   0.000661,   0.000331,   0.000331,   0.000331,
                    0.000413,   0.004298,   0.000165,   0.000165,   0.000661,
                    0.000661,   0.003554,   0.004132,   0.001488,   0.000744,
                    0.004380,   0.000331,   0.004876,   0.000636,   0.033471,
                    0.029587,   0.033058,   0.048016,   0.013802,   0.056364,
                    0.058934,   0.021744,   0.034711,   0.019711,   0.038430,
                    0.024860,   0.022628,   0.022281,   0.025207,   0.020661,
                    0.059669,   0.027107,   0.033661,   0.024132,   0.018347,
                    0.024967,   0.022355,   0.016537,   0.026793,   0.014446,
                    0.020421,   0.032041,   0.016124,   0.016603,   0.104132,
                    0.033058,   0.100000,   0.015703,   0.033802,   0.015703,
                    0.028934,   0.000554,   0.011983,   0.002661,   0.002661,
                    0.003314,   0.009917,   0.025620,   0.004132,   0.018182,
                    0.004132,   0.008769,   0.004446,   0.004967,   0.015703,
                    0.007438,   0.015703,   0.007438,   0.001653,   0.009091,
                    0.009091,   0.009091,   0.023141,   0.013719,   0.022314,
                    0.024893,   0.013620,   0.029496,   0.007521,   0.062149,
                    0.039669,   0.004959,   0.014380,   0.038760,   0.011653,
                    0.011570,   0.007942,   0.003314,   0.009091,   0.011570,
                    0.011570,   0.000826,   0.015231,   0.011570,   0.007066,
                    0.025620,   0.021488,   0.009917,   0.019826,   0.025612,
                    0.005000,   0.028099,   0.023000,   0.025463,   0.037711,
                    0.028959,   0.080661,   0.041157,   0.058083,   0.025041,
                    0.016612,   0.015703,   0.024793,   0.038909,   0.001025,
                    0.032372,   0.002917,   0.006033,   0.042066,   0.029917,
                    0.011653,   0.003719,   0.005372,   0.004132,   0.000744,
                    0.007107,   0.004992,   0.019174,   0.003802,   0.001157,
                    0.014554,   0.020083,   0.017107,   0.031959,   0.018479,
                    0.029008,   0.016612,   0.007934,   0.008636,   0.037603,
                    0.014132,   0.010000,   0.006099,   0.022132,   0.046777,
                    0.053876,   0.041397,   0.009678,   0.011653,   0.022041,
                    0.012141,   0.040331,   0.033636,   0.016686,   0.057603,
                    0.000008,   0.035289,   0.068083,   0.004298,   0.016124,
                    0.039669,   0.023967,   0.046446,   0.022760,   0.011240,
                    0.023471,   0.009826,   0.013934,   0.006876,   0.034298,
                    0.003893,   0.060496,   0.006149,   0.007587,   0.005190,
                    0.067438,   0.014521,   0.001818,   0.002587,   0.001901,
                    0.001744,   0.006860,   0.017017,   0.010281,   0.011256,
                    0.003033,   0.006174,   0.000041,   0.007314,   0.009587,
                    0.009587,   0.013438,   0.006983,   0.005744,   0.003388,
                    0.002810,   0.014752,   0.004050,   0.010314,   0.039504,
                    0.001653,   0.001818,   0.004050,   0.020347,   0.022719,
                    0.001289,   0.028331,   0.029669,   0.028256,   0.017289,
                    0.007017,   0.023876,   0.003554,   0.027372,   0.017917,
                    0.037934,   0.013223,   0.005041,   0.037438,   0.020174,
                    0.015289,   0.033314,   0.038141,   0.000686,   0.000083,
                    0.013636,   0.008678,   0.005372,   0.028430,   0.028430,
                    0.003876,   0.007603,   0.011686,   0.003967,   0.000992,
                    0.026760,   0.000504,   0.000504,   0.002760,   0.013512,
                    0.009091,   0.031983,   0.006917,   0.005041,   0.013950,
                    0.016570,   0.003802,   0.001570,   0.017934,   0.064380,
                    0.002397,   0.014711,   0.025041,   0.016612,   0.014380,
                    0.036686,   0.029826,   0.003017,   0.001612,   0.016446,
                    0.010331,   0.022893,   0.017521,   0.014628,   0.022066,
                    0.017603,   0.016223,   0.009752,   0.034876,   0.006529,
                    0.013479,   0.026198,   0.046686,   0.033967,   0.027364,
                    0.027190,   0.029669,   0.025207,   0.059446,   0.018769,
                    0.021107,   0.024727,   0.053826,   0.002570,   0.001397,
                    0.026612,   0.021984,   0.032240,   0.001975,   0.069529,
                    0.031488,   0.010397,   0.003752,   0.007107,   0.004149,
                    0.006521,   0.019826,   0.010934,   0.003752,   0.003826,
                    0.011727,   0.040231,   0.028603,   0.000091,   0.005554,
                    0.025603,   0.006612,   0.009603,   0.001711,   0.006281,
                    0.017769,   0.027058,   0.020777,   0.041521,   0.050174,
                    0.044628,   0.024793,   0.037686,   0.065372,   0.011438,
                    0.020579,   0.013140,   0.019471,   0.021818,   0.004860,
                    0.002281,   0.023058,   0.003893,   0.010107,   0.007802,
                    0.003174,   0.005950,   0.012893,   0.033967,   0.069256,
                    0.039826,   0.007521,   0.007521,   0.018182,   0.009256,
                    0.069421,   0.011066,   0.012893,   0.008017,   0.008430,
                    0.022636,   0.031802,   0.050331,   0.028512,   0.034298,
                    0.008182,   0.048760,   0.019793,   0.023636,   0.001818,
                    0.004050,   0.022314,   0.028347,   0.030248,   0.015703,
                    0.012008,   0.011273,   0.008099,   0.029752,   0.049587,
                    0.042066,   0.038983,   0.014669,   0.051240,   0.006612,
                    0.014050,   0.024793,   0.032231,   0.017620,   0.011347,
                    0.009917,   0.017529,   0.009917,   0.033802,   0.028099,
                    0.014050,   0.018182,   0.011570,   0.024793,   0.037934,
                    0.038529,   0.025149,   0.014050,   0.008256,   0.037636,
                    0.016760,   0.013347,   0.013173,   0.042827,   0.054893,
                    0.049016,   0.005141,   0.005141,   0.032231,   0.027273,
                    0.003314,   0.004132,   0.004132,   0.004132,   0.004132,
                    0.010744,   0.009446,   0.004512,   0.006504,   0.013190,
                    0.012603,   0.025620,   0.012397,   0.036686,   0.006612,
                    0.005917,   0.001570,   0.001529,   0.005793,   0.027190,
                    0.012901,   0.012041,   0.028934,   0.015703,   0.018934,
                    0.014050,   0.009000,   0.009091,   0.012397,   0.006364,
                    0.010281,   0.013339,   0.012488,   0.000603,   0.004347,
                    0.000397,   0.002207,   0.008504,   0.008198,   0.002570,
                    0.004554,   0.002570,   0.011760,   0.002570,   0.018033,
                    0.001537,   0.002479,   0.002479,   0.006967,   0.002769,
                    0.002769,   0.000124,   0.000099,   0.005793,   0.020702,
                    0.003488,   0.006149,   0.024793,   0.028934,   0.006760,
                    0.003686,   0.005223,   0.000612,   0.001017,   0.004314,
                    0.009421,   0.006446,   0.024388,   0.019016,   0.005793,
                    0.015703,   0.008256,   0.022091,   0.004347,   0.004347,
                    0.046281,   0.002446,   0.002446,   0.006612,   0.007438,
                    0.004959,   0.004959,   0.013223,   0.009091,   0.009917,
                    0.024793,   0.014992,   0.042984,   0.002868,   0.037190,
                    0.064628,   0.033223,   0.017826,   0.007587,   0.012091,
                    0.006868,   0.005529,   0.015703,   0.004132,   0.065826,
                    0.020661,   0.016421,   0.012141,   0.031240,   0.044207,
                    0.010331,   0.012562,   0.010083,   0.020000,   0.007934,
                    0.010826,   0.028182,   0.057016,   0.039339,   0.005950,
                    0.036198,   0.027430,   0.040331,   0.011570,   0.035066,
                    0.041066,   0.029496,   0.011322,   0.014207,   0.017769,
                    0.022686,   0.020992,   0.017107,   0.024959,   0.031744,
                    0.025950,   0.009430,   0.017686,   0.052339,   0.035620,
                    0.036777,   0.002066,   0.018512,   0.021000,   0.033471,
                    0.066446,   0.039835,   0.033802,   0.020496,   0.040000,
                    0.009917,   0.040248,   0.035446,   0.012636,   0.033058,
                    0.046686,   0.034132,   0.035868,   0.100000,   0.039504,
                    0.031240,   0.021397,   0.016859,   0.000256,   0.012231,
                    0.028827,   0.018240,   0.024554,   0.068430,   0.086529,
                    0.007851,   0.010331,   0.022314,   0.030992,   0.002636,
                    0.030248,   0.003314,   0.041240,   0.018603,   0.024793,
                    0.013140,   0.009421,   0.029008,   0.004207,   0.007364,
                    0.001570,   0.001570,   0.003058,   0.008256,   0.018099,
                    0.004050,   0.002314,   0.003967,   0.008099,   0.064711,
                    0.066612,   0.013223,   0.056860,   0.037603,   0.030579,
                    0.008512,   0.040496,   0.001983,   0.006529,   0.009256,
                    0.062231,   0.047017,   0.005446,   0.016198,   0.007917,
                    0.027810,   0.013967,   0.008678,   0.016066,   0.034000,
                    0.005620,   0.005620,   0.002149,   0.000909,   0.006612,
                    0.020496,   0.037397,   0.019281,   0.004959,   0.001653,
                    0.057686,   0.031983,   0.018430,   0.005537,   0.036364,
                    0.003058,   0.067603,   0.038678,   0.050422,   0.019760,
                    0.027810,   0.014669,   0.022314,   0.013223,   0.008835,
                    0.006876,   0.015537,   0.008512,   0.002810,   0.027686,
                    0.021322,   0.015041,   0.017769,   0.007364,   0.004207,
                    0.023314,   0.002727,   0.001488,   0.028678,   0.013636,
                    0.004207,   0.005289,   0.009339,   0.034793,   0.021240,
                    0.031397,   0.008512,   0.018099,   0.017190,   0.015793,
                    0.033636,   0.008256,   0.004959,   0.004959,   0.004959,
                    0.004959,   0.004959,   0.025405,   0.016364,   0.020413,
                    0.009917,   0.020174,   0.009504,   0.014050,   0.032860,
                    0.042231,   0.073058,   0.030909,   0.042066,   0.016529,
                    0.040496,   0.046281,   0.004959,   0.004132,   0.013364,
                    0.008678,   0.044380,   0.056612,   0.019752,   0.036686,
                    0.035868,   0.034380,   0.035868,   0.020397,   0.014554,
                    0.006612,   0.009826,   0.030248,   0.011488,   0.008256,
                    0.001240,   0.001240,   0.001240,   0.010826,   0.001240,
                    0.001240,   0.001240,   0.036686,   0.017364,   0.032479,
                    0.054554,   0.019174,   0.014050,   0.035868,   0.015000,
                    0.005248,   0.000256,   0.041744,   0.023058,   0.000992,
                    0.003636,   0.004628,   0.003719,   0.004959,   0.011157,
                    0.011157,   0.014959,   0.007603,   0.034793,   0.007017,
                    0.008430,   0.008430,   0.019917,   0.011868,   0.005041,
                    0.003314,   0.007769,   0.007107,   0.003058,   0.006446,
                    0.001240,   0.001240,   0.020744,   0.005446,   0.006124,
                    0.001322,   0.007934,   0.009091,   0.007364,   0.003314,
                    0.002066,   0.011322,   0.002893,   0.002479,   0.004917,
                    0.004917,   0.001653,   0.039504,   0.039504,   0.053058,
                    0.032934,   0.058066,   0.024438,   0.021388,   0.001240,
                    0.004628,   0.004628,   0.012810,   0.026198,   0.038843,
                    0.011653,   0.021397,   0.002810,   0.158017,   0.009826,
                    0.010388,   0.036033,   0.022339,   0.041397,   0.033141,
                    0.037686,   0.019422,   0.035868,   0.030248,   0.009826,
                    0.053223,   0.006612,   0.005372,   0.023388,   0.011570,
                    0.033099,   0.028182,   0.024463,   0.006612,   0.006612,
                    0.020496,   0.058182,   0.012810,   0.012810,   0.014463,
                    0.023554,   0.027603,   0.039669,   0.029826,   0.009256,
                    0.053141,   0.001322,   0.009091,   0.055041,   0.036529,
                    0.015950,   0.007438,   0.036686,   0.083141,   0.026612,
                    0.033471,   0.005207,   0.018430,   0.001818,   0.003471,
                    0.037603,   0.019669,   0.003314,   0.029752,   0.005124,
                    0.033471,   0.016529,   0.033471,   0.016529,   0.022884,
                    0.001653,   0.009091,   0.004050,   0.020744,   0.028347,
                    0.040331,   0.020413,   0.025620,   0.025620,   0.010413,
                    0.009752,   0.002562,   0.003058,   0.000248,   0.005702,
                    0.000021,   0.000021,   0.000006,   0.000021,   0.000021,
                    0.000006,   0.000021,   0.000021,   0.000021,   0.000006,
                    0.000021,   0.000021,   0.000021,   0.000006,   0.000006,
                    0.000021,   0.000006,   0.000021,   0.000006,   0.000021,
                    0.000021,   0.000006,   0.000021,   0.000021,   0.000021,
                    0.000021,   0.000021,   0.000021,   0.000021,   0.000021,
                    0.000021,   0.000006,   0.000021,   0.000021,   0.000021,
                    0.000021,   0.000021,   0.000021,   0.000021,   0.000021,
                    0.000021,   0.000021,   0.000021,   0.000021,   0.000006,
                    0.000006,   0.000021,   0.000006,   0.000021,   0.000021,
                    0.000006,   0.000021,   0.000021,   0.000021,   0.000021,
                    0.000006,   0.000021,   0.000021,   0.000006,   0.000021,
                    0.000006,   0.000021,   0.000021,   0.000021,   0.000006,
                    0.000021,   0.000021,   0.000021,   0.000006,   0.000006,
                    0.000021,   0.000021,   0.000021,   0.000006,   0.000021,
                    0.000006,   0.000021,   0.000021,   0.000006,   0.000021,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000000,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000000,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000000,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000000,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000000,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000000,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000000,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000000,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000000,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000000,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083,   0.000083,
                    0.000083,   0.000083,   0.000083,   0.000083])

    X_br = np.array([0.030500,   0.025190,   0.025190,   0.020970,   0.020970,
                    0.030000,   0.024440,   0.024690,   0.028750,   0.030500,
                    0.023000,   0.023000,   0.023750,   0.030930,   0.030000,
                    0.003380,   0.003380,   0.083531,   0.070659,   0.070870,
                    0.016837,   0.015909,   0.015909,   0.022870,   0.022870,
                    0.031162,   0.081141,   0.062401,   0.064050,   0.123659,
                    0.026180,   0.061961,   0.105891,   0.081610,   0.072310,
                    0.059484,   0.068289,   0.089091,   0.087025,   0.097169,
                    0.074979,   0.104130,   0.043800,   0.020560,   0.015060,
                    0.034630,   0.019940,   0.016250,   0.027690,   0.024740,
                    0.017880,   0.055440,   0.034250,   0.023690,   0.097110,
                    0.040700,   0.032229,   0.112190,   0.087419,   0.050516,
                    0.078950,   0.039521,   0.039521,   0.039880,   0.000207,
                    0.006510,   0.013260,   0.043000,   0.015990,   0.011690,
                    0.011690,   0.028531,   0.034180,   0.042620,   0.039770,
                    0.036750,   0.013290,   0.009000,   0.060969,   0.065641,
                    0.065581,   0.070310,   0.040211,   0.040211,   0.048680,
                    0.020760,   0.051961,   0.027831,   0.057271,   0.041450,
                    0.033680,   0.008781,   0.042831,   0.045409,   0.062151,
                    0.017979,   0.014940,   0.018900,   0.012770,   0.026439,
                    0.010970,   0.014060,   0.021440,   0.033616,   0.059194,
                    0.013926,   0.029752,   0.004959,   0.040393,   0.005248,
                    0.087603,   0.014587,   0.116777,   0.040930,   0.028471,
                    0.042417,   0.042417,   0.038388,   0.014029,   0.027748,
                    0.070744,   0.041467,   0.012789,   0.009731,   0.004112,
                    0.006240,   0.006260,   0.011116,   0.021591,   0.031570,
                    0.009917,   0.026674,   0.041281,   0.030413,   0.035062,
                    0.023409,   0.012603,   0.009607,   0.024421,   0.012603,
                    0.041488,   0.024628,   0.013327,   0.004876,   0.004876,
                    0.027544,   0.024319,   0.010400,   0.035631,   0.010512,
                    0.073349,   0.073349,   0.079550,   0.079550,   0.044010,
                    0.080570,   0.042310,   0.095599,   0.053000,   0.091359,
                    0.036539,   0.077891,   0.056917,   0.016674,   0.089554,
                    0.013180,   0.025930,   0.076219,   0.052250,   0.032521,
                    0.031880,   0.003310,   0.003310,   0.076859,   0.034651,
                    0.041300,   0.007211,   0.044831,   0.016550,   0.036979,
                    0.017789,   0.026960,   0.017330,   0.015740,   0.043260,
                    0.137581,   0.160140,   0.111320,   0.059190,   0.021141,
                    0.055789,   0.041961,   0.042360,   0.015480,   0.025830,
                    0.024750,   0.017680,   0.015690,   0.015690,   0.066120,
                    0.066120,   0.047273,   0.036780,   0.067360,   0.067360,
                    0.069010,   0.069010,   0.068599,   0.068599,   0.061880,
                    0.067750,   0.067750,   0.067750,   0.052440,   0.052440,
                    0.068391,   0.068391,   0.069010,   0.069010,   0.066120,
                    0.066120,   0.069010,   0.069010,   0.067979,   0.069010,
                    0.061880,   0.068180,   0.061190,   0.061190,   0.069010,
                    0.069010,   0.069010,   0.066800,   0.066800,   0.067151,
                    0.078081,   0.066159,   0.067750,   0.069461,   0.060520,
                    0.060520,   0.061560,   0.066159,   0.062350,   0.059840,
                    0.068391,   0.067750,   0.067750,   0.067831,   0.068310,
                    0.060800,   0.062350,   0.067539,   0.067620,   0.067771,
                    0.067979,   0.069211,   0.062690,   0.062690,   0.066940,
                    0.066531,   0.065500,   0.070450,   0.069831,   0.067771,
                    0.067360,   0.066940,   0.068391,   0.066940,   0.066940,
                    0.067560,   0.067560,   0.066531,   0.068599,   0.093599,
                    0.092771,   0.067151,   0.065700,   0.067560,   0.067560,
                    0.062690,   0.062940,   0.067360,   0.066740,   0.059880,
                    0.061190,   0.062690,   0.066940,   0.068599,   0.068180,
                    0.067560,   0.065500,   0.062500,   0.068180,   0.068180,
                    0.066880,   0.062060,   0.062060,   0.066880,   0.066880,
                    0.066531,   0.066531,   0.068180,   0.068180,   0.068510,
                    0.231401,   0.452479,   0.226240,   0.061359,   0.059300,
                    0.059300,   0.059300,   0.092771,   0.066940,   0.061359,
                    0.068599,   0.061880,   0.061880,   0.059310,   0.070331,
                    0.059300,   0.059300,   0.106010,   0.106010,   0.059300,
                    0.061880,   0.062060,   0.066531,   0.067750,   0.052289,
                    0.068180,   0.068180,   0.066320,   0.066320,   0.066320,
                    0.065847,   0.059687,   0.064369,   0.064369,   0.186570,
                    0.066942,   0.066715,   0.066715,   0.042063,   0.067355,
                    0.067355,   0.054113,   0.067562,   0.067562,   0.042063,
                    0.106198,   0.067500,   0.067355,   0.066715,   0.059687,
                    0.059687,   0.067975,   0.067748,   0.101322,   0.061240,
                    0.062810,   0.108603,   0.105041,   0.103967,   0.080992,
                    0.071066,   0.083876,   0.130248,   0.090083,   0.079339,
                    0.060331,   0.139669,   0.053719,   0.049091,   0.066124,
                    0.120661,   0.027273,   0.184298,   0.060744,   0.051157,
                    0.096686,   0.200083,   0.115207,   0.087603,   0.107438,
                    0.183223,   0.064463,   0.074380,   0.323967,   0.050413,
                    0.113223,   0.053141,   0.130504,   0.029752,   0.089256,
                    0.045041,   0.101488,   0.130248,   0.126446,   0.067190,
                    0.055703,   0.042984,   0.040496,   0.033876,   0.062810,
                    0.075289,   0.130413,   0.095041,   0.185868,   0.096686,
                    0.071901,   0.058678,   0.099173,   0.067603,   0.037190,
                    0.028678,   0.063554,   0.022066,   0.018017,   0.009496,
                    0.061066,   0.061901,   0.044876,   0.059587,   0.058512,
                    0.061240,   0.007769,   0.076124,   0.043967,   0.071570,
                    0.059917,   0.079669,   0.121653,   0.190744,   0.100083,
                    0.064554,   0.068760,   0.077364,   0.008017,   0.082810,
                    0.074463,   0.095124,   0.044554,   0.076859,   0.103471,
                    0.083141,   0.079752,   0.020248,   0.121322,   0.073223,
                    0.104554,   0.082636,   0.155041,   0.078099,   0.022810,
                    0.044380,   0.052727,   0.030413,   0.019339,   0.014711,
                    0.019339,   0.019917,   0.010826,   0.020661,   0.020661,
                    0.017769,   0.019008,   0.026942,   0.033967,   0.027190,
                    0.009421,   0.025289,   0.005124,   0.006686,   0.048016,
                    0.039669,   0.010578,   0.010578,   0.017603,   0.003058,
                    0.006942,   0.044463,   0.015793,   0.034711,   0.028256,
                    0.005289,   0.019587,   0.032149,   0.001653,   0.007207,
                    0.003545,   0.039174,   0.014132,   0.035620,   0.001488,
                    0.011653,   0.005868,   0.016942,   0.012810,   0.001901,
                    0.007934,   0.009917,   0.001744,   0.010661,   0.024628,
                    0.022314,   0.018934,   0.006860,   0.041322,   0.011397,
                    0.001488,   0.006942,   0.007851,   0.009917,   0.029752,
                    0.038760,   0.053554,   0.008512,   0.022066,   0.012314,
                    0.063554,   0.040174,   0.027851,   0.066446,   0.003719,
                    0.013314,   0.002314,   0.002314,   0.010992,   0.012727,
                    0.107851,   0.069752,   0.120000,   0.076281,   0.037190,
                    0.031240,   0.095868,   0.030752,   0.019669,   0.029826,
                    0.025446,   0.020909,   0.039256,   0.139339,   0.011570,
                    0.082727,   0.014207,   0.191983,   0.009256,   0.012984,
                    0.055124,   0.019339,   0.030413,   0.101240,   0.014463,
                    0.012066,   0.128603,   0.055868,   0.135124,   0.127686,
                    0.064050,   0.108182,   0.010909,   0.011322,   0.068430,
                    0.017364,   0.057438,   0.042984,   0.029174,   0.075207,
                    0.037934,   0.022562,   0.011397,   0.020744,   0.022810,
                    0.064793,   0.084628,   0.017521,   0.056612,   0.088099,
                    0.071570,   0.044711,   0.017521,   0.017769,   0.034207,
                    0.018512,   0.007273,   0.009587,   0.016612,   0.046364,
                    0.014876,   0.022727,   0.002810,   0.026364,   0.013876,
                    0.011901,   0.001983,   0.009917,   0.016446,   0.023876,
                    0.006446,   0.018182,   0.012397,   0.017934,   0.012984,
                    0.022636,   0.036281,   0.047017,   0.016124,   0.028017,
                    0.008603,   0.000331,   0.011240,   0.006777,   0.032314,
                    0.047851,   0.013719,   0.004298,   0.051984,   0.046686,
                    0.010174,   0.004298,   0.008256,   0.024711,   0.008182,
                    0.014463,   0.025372,   0.030579,   0.041157,   0.019752,
                    0.038603,   0.018099,   0.072149,   0.012562,   0.014959,
                    0.015207,   0.088678,   0.082231,   0.004711,   0.021397,
                    0.035446,   0.026124,   0.037686,   0.046529,   0.030388,
                    0.053876,   0.032727,   0.073314,   0.012562,   0.043636,
                    0.079422,   0.026281,   0.061818,   0.013471,   0.085124,
                    0.132636,   0.008430,   0.037273,   0.012479,   0.011744,
                    0.024471,   0.094711,   0.106033,   0.007851,   0.038934,
                    0.034793,   0.031983,   0.042149,   0.009421,   0.024207,
                    0.044463,   0.011116,   0.063802,   0.011570,   0.093141,
                    0.022149,   0.072479,   0.047686,   0.063314,   0.066942,
                    0.027934,   0.015868,   0.027686,   0.137017,   0.078934,
                    0.012727,   0.012066,   0.053149,   0.007190,   0.022727,
                    0.000083,   0.063141,   0.117190,   0.093471,   0.062479,
                    0.023223,   0.039339,   0.094380,   0.059339,   0.064380,
                    0.079339,   0.072810,   0.026198,   0.025446,   0.030000,
                    0.054050,   0.064380,   0.018678,   0.053876,   0.034959,
                    0.087438,   0.062727,   0.064959,   0.082231,   0.022984,
                    0.074132,   0.127603,   0.060248,   0.099669,   0.029504,
                    0.063802,   0.071570,   0.006198,   0.017107,   0.010000,
                    0.030992,   0.034628,   0.012984,   0.159008,   0.104207,
                    0.016033,   0.030000,   0.022149,   0.042984,   0.037769,
                    0.041157,   0.145289,   0.074298,   0.072397,   0.133719,
                    0.031570,   0.066942,   0.101240,   0.008182,   0.022562,
                    0.116281,   0.177364,   0.090000,   0.087190,   0.091240,
                    0.019917,   0.081901,   0.033802,   0.082149,   0.089669,
                    0.023876,   0.117107,   0.012149,   0.042727,   0.063141,
                    0.008512,   0.069091,   0.103314,   0.037364,   0.095372,
                    0.070661,   0.125702,   0.047364,   0.084463,   0.080496,
                    0.083636,   0.061983,   0.119008,   0.101653,   0.012397,
                    0.154793,   0.094959,   0.074380,   0.040083,   0.056777,
                    0.080578,   0.013876,   0.120331,   0.025446,   0.071570,
                    0.055372,   0.088182,   0.080992,   0.065950,   0.084959,
                    0.071488,   0.007521,   0.004231,   0.034405,   0.000744,
                    0.030083,   0.000083,   0.094207,   0.098934,   0.086529,
                    0.072893,   0.149752,   0.138099,   0.109091,   0.027769,
                    0.053058,   0.132893,   0.071570,   0.071570,   0.065289,
                    0.127769,   0.103471,   0.126686,   0.124628,   0.198603,
                    0.008843,   0.084380,   0.038934,   0.065537,   0.028198,
                    0.032967,   0.070190,   0.046033,   0.009917,   0.007934,
                    0.008256,   0.021653,   0.020248,   0.082810,   0.094793,
                    0.049256,   0.007603,   0.009587,   0.060992,   0.081066,
                    0.013223,   0.066281,   0.025124,   0.011570,   0.019174,
                    0.008017,   0.003967,   0.010248,   0.107521,   0.044132,
                    0.026860,   0.015793,   0.040909,   0.063636,   0.091397,
                    0.070661,   0.098760,   0.018603,   0.107364,   0.206124,
                    0.044380,   0.028603,   0.024132,   0.030661,   0.012984,
                    0.104380,   0.130579,   0.012810,   0.042066,   0.098099,
                    0.122983,   0.031157,   0.009587,   0.085207,   0.062314,
                    0.099091,   0.100413,   0.061983,   0.080744,   0.016942,
                    0.035793,   0.065620,   0.013388,   0.008017,   0.055041,
                    0.012397,   0.055041,   0.051157,   0.127521,   0.044876,
                    0.100413,   0.032983,   0.009008,   0.021901,   0.046612,
                    0.113802,   0.144132,   0.122727,   0.030826,   0.033314,
                    0.078843,   0.100744,   0.121901,   0.099504,   0.073636,
                    0.018934,   0.027851,   0.030504,   0.018347,   0.089752,
                    0.037893,   0.076884,   0.027016,   0.059504,   0.066860,
                    0.157686,   0.033058,   0.086033,   0.056777,   0.121397,
                    0.115041,   0.053802,   0.031818,   0.005372,   0.019174,
                    0.029587,   0.025703,   0.148678,   0.026942,   0.085314,
                    0.101066,   0.112810,   0.006612,   0.037107,   0.055703,
                    0.020744,   0.120248,   0.079256,   0.024554,   0.089256,
                    0.071240,   0.014554,   0.006942,   0.052231,   0.031744,
                    0.019752,   0.019091,   0.011901,   0.073140,   0.235124,
                    0.019917,   0.014793,   0.024050,   0.010496,   0.087769,
                    0.031488,   0.080909,   0.046942,   0.019256,   0.076446,
                    0.072397,   0.065950,   0.132562,   0.079587,   0.022893,
                    0.075620,   0.071397,   0.059173,   0.067364,   0.036446,
                    0.067017,   0.026198,   0.020000,   0.051066,   0.005620,
                    0.036281,   0.090496,   0.015372,   0.106860,   0.106686,
                    0.066777,   0.047686,   0.013140,   0.014628,   0.008182,
                    0.016859,   0.125950,   0.047438,   0.066364,   0.031818,
                    0.078017,   0.057364,   0.026446,   0.054132,   0.027438,
                    0.020579,   0.058016,   0.002636,   0.001901,   0.092810,
                    0.050744,   0.068934,   0.026529,   0.029669,   0.020000,
                    0.058512,   0.016033,   0.049826,   0.074628,   0.000504,
                    0.059314,   0.043942,   0.131397,   0.025620,   0.014380,
                    0.087603,   0.060000,   0.044463,   0.051901,   0.057438,
                    0.009917,   0.057364,   0.066124,   0.031066,   0.083223,
                    0.075207,   0.014876,   0.025793,   0.133388,   0.111157,
                    0.077686,   0.007769,   0.061818,   0.065620,   0.049752,
                    0.007107,   0.123802,   0.067934,   0.021818,   0.034298,
                    0.101397,   0.007934,   0.007934,   0.000579,   0.052727,
                    0.112562,   0.051322,   0.051818,   0.018099,   0.055124,
                    0.043223,   0.100826,   0.070413,   0.009826,   0.014554,
                    0.045372,   0.024554,   0.013876,   0.018512,   0.068430,
                    0.028430,   0.066033,   0.020174,   0.031570,   0.090579,
                    0.046612,   0.059587,   0.018512,   0.085207,   0.016859,
                    0.029826,   0.006281,   0.010331,   0.012397,   0.012984,
                    0.011570,   0.057603,   0.057603,   0.038347,   0.055620,
                    0.055620,   0.067273,   0.003058,   0.067521,   0.072231,
                    0.049008,   0.011901,   0.009008,   0.074132,   0.062636,
                    0.014711,   0.064876,   0.067851,   0.035124,   0.059587,
                    0.082975,   0.009669,   0.050413,   0.000083,   0.049174,
                    0.000248,   0.050413,   0.001488,   0.099587,   0.017190,
                    0.095620,   0.000248,   0.046694,   0.094215,   0.001157,
                    0.046694,   0.094215,   0.001157,   0.119835,   0.105785,
                    0.035950,   0.035950,   0.008926,   0.023471,   0.000413,
                    0.044628,   0.063636,   0.018017,   0.048265,   0.022231,
                    0.013388,   0.017686,   0.018430,   0.002893,   0.015372,
                    0.031405,   0.008595,   0.015372,   0.031405,   0.008512,
                    0.070165,   0.065620,   0.055372,   0.045868,   0.019752,
                    0.008430,   0.019504,   0.015454,   0.008430,   0.045207,
                    0.045207,   0.008347,   0.015620,   0.016446,   0.004463,
                    0.017355,   0.060826,   0.011983,   0.048016,   0.065454,
                    0.007273,   0.007355,   0.031901,   0.041322,   0.023967,
                    0.024959,   0.083471,   0.093058,   0.014628,   0.074711,
                    0.051984,   0.050826,   0.051240,   0.065702,   0.070579,
                    0.060579,   0.023719,   0.060165,   0.129917,   0.079091,
                    0.033388,   0.026364,   0.063884,   0.003141,   0.040992,
                    0.017025,   0.067934,   0.119504,   0.059339,   0.053554,
                    0.021488,   0.048926,   0.021901,   0.021901,   0.227273,
                    0.211240,   0.105950,   0.000909,   0.007107,   0.006942,
                    0.002149,   0.000248,   0.050165,   0.061488,   0.041322,
                    0.061735,   0.056860,   0.009504,   0.043306,   0.024793,
                    0.039752,   0.000826,   0.010992,   0.000248,   0.004380,
                    0.006777,   0.002314,   0.021818,   0.000744,   0.002149,
                    0.007934,   0.007934,   0.010992,   0.011240,   0.011240,
                    0.067190,   0.028099,   0.011735,   0.001736,   0.007769,
                    0.014545,   0.033058,   0.057769,   0.015703,   0.032727,
                    0.026033,   0.065950,   0.026033,   0.047686,   0.025950,
                    0.008430,   0.036116,   0.026777,   0.063223,   0.102397,
                    0.115950,   0.076942,   0.041818,   0.000826,   0.030000,
                    0.032231,   0.013471,   0.104959,   0.031983,   0.009504,
                    0.000826,   0.000826,   0.013967,   0.006033,   0.017521,
                    0.017521,   0.060413,   0.015289,   0.025041,   0.068265,
                    0.052066,   0.030248,   0.007438,   0.042231,   0.004793,
                    0.031322,   0.025041,   0.015041,   0.007934,   0.014215,
                    0.007025,   0.042645,   0.033802,   0.012479,   0.026529,
                    0.009008,   0.032314,   0.051488,   0.008347,   0.061570,
                    0.023884,   0.008347,   0.051984,   0.053884,   0.020661,
                    0.076198,   0.046198,   0.090165,   0.008017,   0.031240,
                    0.009835,   0.076859,   0.024711,   0.024711,   0.061983,
                    0.013058,   0.011653,   0.065372,   0.015868,   0.050248,
                    0.024711,   0.009256,   0.070909,   0.021570,   0.028017,
                    0.014959,   0.044876,   0.042645,   0.000248,   0.014132,
                    0.070579,   0.040579,   0.018678,   0.001157,   0.003636,
                    0.000496,   0.006612,   0.045041,   0.025124,   0.020909,
                    0.000331,   0.000992,   0.013636,   0.043388,   0.063967,
                    0.017273,   0.013967,   0.017190,   0.033967,   0.044050,
                    0.013471,   0.002066,   0.045207,   0.002479,   0.003884,
                    0.010661,   0.000826,   0.004876,   0.028760,   0.015868,
                    0.003802,   0.002562,   0.025537,   0.000661,   0.024380,
                    0.025372,   0.043967,   0.031901,   0.011901,   0.006612,
                    0.030744,   0.003058,   0.018430,   0.002397,   0.036446,
                    0.013306,   0.015950,   0.014628,   0.006116,   0.000165,
                    0.004132,   0.004628,   0.021984,   0.020661,   0.001488,
                    0.002893,   0.005289,   0.041074,   0.012149,   0.021653,
                    0.014463,   0.002562,   0.003802,   0.014297,   0.010744,
                    0.004463,   0.007273,   0.011570,   0.056777,   0.002893,
                    0.004463,   0.013388,   0.003884,   0.000826,   0.008347,
                    0.007603,   0.000165,   0.009669,   0.000826,   0.003388,
                    0.013884,   0.001488,   0.003388,   0.011901,   0.001901,
                    0.025289,   0.014545,   0.009339,   0.039669,   0.000248,
                    0.000413,   0.001488,   0.001488,   0.038265,   0.003388,
                    0.001322,   0.026942,   0.012066,   0.002562,   0.006364,
                    0.001322,   0.004876,   0.013223,   0.002149,   0.037107,
                    0.013967,   0.009752,   0.015041,   0.002645,   0.002645,
                    0.007025,   0.007025,   0.009008,   0.014380,   0.006116,
                    0.023471,   0.004793,   0.022562,   0.003306,   0.001405,
                    0.000496,   0.026529,   0.024546,   0.029669,   0.006116,
                    0.024546,   0.021074,   0.005372,   0.007355,   0.013306,
                    0.003802,   0.003884,   0.003223,   0.005620,   0.001653,
                    0.021405,   0.015008,   0.012397,   0.006033,   0.015207,
                    0.001818,   0.007438,   0.018017,   0.000248,   0.004628,
                    0.004628,   0.022645,   0.001240,   0.023058,   0.001157,
                    0.007521,   0.026116,   0.011405,   0.004876,   0.006942,
                    0.005702,   0.017107,   0.015207,   0.001818,   0.016281,
                    0.041488,   0.024050,   0.010992,   0.027851,   0.034463,
                    0.007025,   0.010826,   0.006033,   0.016777,   0.009835,
                    0.005620,   0.002314,   0.008926,   0.019174,   0.008926,
                    0.004463,   0.016033,   0.001074,   0.023884,   0.026446,
                    0.012314,   0.076281,   0.007769,   0.016116,   0.006446,
                    0.000496,   0.023967,   0.038678,   0.003719,   0.051653,
                    0.023967,   0.001818,   0.007438,   0.008512,   0.010496,
                    0.016942,   0.008926,   0.061488,   0.033471,   0.005537,
                    0.014628,   0.015785,   0.024050,   0.000826,   0.002314,
                    0.021736,   0.003141,   0.028265,   0.031157,   0.041570,
                    0.036612,   0.031901,   0.006777,   0.000083,   0.016364,
                    0.014463,   0.094876,   0.032810,   0.016694,   0.003802,
                    0.020909,   0.037851,   0.060992,   0.013140,   0.019504,
                    0.013802,   0.031157,   0.030826,   0.004298,   0.023554,
                    0.025537,   0.004628,   0.019752,   0.019587,   0.002479,
                    0.002397,   0.005289,   0.010413,   0.019669,   0.014711,
                    0.027603,   0.015537,   0.008760,   0.046364,   0.015868,
                    0.010496,   0.055289,   0.016033,   0.027355,   0.027603,
                    0.026198,   0.026198,   0.013140,   0.013140,   0.012397,
                    0.032231,   0.013058,   0.044711,   0.020000,   0.013802,
                    0.012810,   0.011157,   0.014297,   0.014297,   0.010992,
                    0.005372,   0.007686,   0.018926,   0.003802,   0.007355,
                    0.007769,   0.009008,   0.020083,   0.000248,   0.019587,
                    0.008430,   0.008512,   0.001322,   0.006777,   0.004959,
                    0.009752,   0.009421,   0.000083,   0.004298,   0.032727,
                    0.026612,   0.031488,   0.019669,   0.006033,   0.020496,
                    0.010661,   0.020909,   0.007355,   0.027190,   0.026116,
                    0.012149,   0.043223,   0.010826,   0.004463,   0.002975,
                    0.013223,   0.025785,   0.004628,   0.006116,   0.014545,
                    0.001983,   0.000413,   0.014545,   0.002727,   0.008595,
                    0.003058,   0.016198,   0.001405,   0.001322,   0.001322,
                    0.001157,   0.015289,   0.017851,   0.027769,   0.018347,
                    0.027273,   0.004380,   0.009256,   0.012645,   0.008347,
                    0.002893,   0.024050,   0.032562,   0.015950,   0.000331,
                    0.005207,   0.129835,   0.056942,   0.200248,   0.003306,
                    0.028512,   0.056694,   0.061322,   0.072645,   0.045207,
                    0.194711,   0.040496,   0.075454,   0.000992,   0.054711,
                    0.037603,   0.027273,   0.025124,   0.022149,   0.022231,
                    0.021074,   0.052149,   0.035537,   0.002314,   0.002314,
                    0.074132,   0.073802,   0.005455,   0.074876,   0.074132,
                    0.009835,   0.005455,   0.066694,   0.068430,   0.061322,
                    0.061322,   0.065454,   0.003471,   0.003471,   0.050661,
                    0.007438,   0.036116,   0.021736,   0.082397,   0.065702,
                    0.069917,   0.014876,   0.102314,   0.082810,   0.062562,
                    0.040413,   0.068512,   0.040083,   0.019835,   0.041736,
                    0.014215,   0.007273,   0.007190,   0.056860,   0.058016,
                    0.054545,   0.044050,   0.096116,   0.034545,   0.108843,
                    0.067686,   0.094380,   0.036364,   0.046529,   0.031240,
                    0.031736,   0.010331,   0.022479,   0.110579,   0.110579,
                    0.074876,   0.022066,   0.053058,   0.150413,   0.089008,
                    0.089008,   0.043967,   0.001653,   0.001653,   0.002066,
                    0.058264,   0.010165,   0.089835,   0.000496,   0.041984,
                    0.062810,   0.089339,   0.025785,   0.043802,   0.010165,
                    0.017438,   0.000579,   0.052562,   0.002397,   0.002645,
                    0.005537,   0.005702,   0.014876,   0.026612,   0.012810,
                    0.006529,   0.006446,   0.006446,   0.002231,   0.002066,
                    0.005124,   0.006364,   0.044380,   0.019091,   0.036364,
                    0.012066,   0.006446,   0.034132,   0.037521,   0.009174,
                    0.004380,   0.006281,   0.057851,   0.029752,   0.010165,
                    0.040248,   0.029174,   0.007438,   0.169835,   0.046694,
                    0.090413,   0.103058,   0.024793,   0.091570,   0.017355,
                    0.099504,   0.034463,   0.012479,   0.070331,   0.007190,
                    0.011818,   0.011653,   0.057934,   0.019835,   0.030331,
                    0.119669,   0.008182,   0.026033,   0.043388,   0.014711,
                    0.020496,   0.018760,   0.018760,   0.041157,   0.023719,
                    0.076446,   0.012231,   0.040083,   0.015289,   0.035289,
                    0.106364,   0.011157,   0.020909,   0.070744,   0.233802,
                    0.237025,   0.002645,   0.007190,   0.131240,   0.012562,
                    0.005793,   0.005793,   0.012149,   0.054793,   0.034380,
                    0.015950,   0.021074,   0.056529,   0.003884,   0.115041,
                    0.104628,   0.164628,   0.035620,   0.008264,   0.029091,
                    0.015785,   0.001983,   0.001818,   0.000579,   0.001322,
                    0.030331,   0.028926,   0.042810,   0.026446,   0.058347,
                    0.000826,   0.019091,   0.010165,   0.001570,   0.018512,
                    0.068430,   0.005868,   0.005041,   0.001818,   0.015620,
                    0.011074,   0.019917,   0.016529,   0.032562,   0.021488,
                    0.011074,   0.024215,   0.037025,   0.017851,   0.092066,
                    0.005124,   0.061322,   0.003884,   0.037686,   0.020579,
                    0.047190,   0.009504,   0.109421,   0.045785,   0.062975,
                    0.011157,   0.006364,   0.006446,   0.030083,   0.005868,
                    0.025950,   0.031570,   0.007934,   0.030579,   0.031570,
                    0.006033,   0.011488,   0.003223,   0.011074,   0.011074,
                    0.029917,   0.022066,   0.003058,   0.003636,   0.002479,
                    0.005455,   0.010413,   0.011735,   0.007190,   0.039008,
                    0.038595,   0.016694,   0.003397,   0.011488,   0.000496,
                    0.016198,   0.007603,   0.027851,   0.020496,   0.000826,
                    0.001240,   0.007686,   0.014132,   0.001074,   0.003884,
                    0.011405,   0.006942,   0.016116,   0.007273,   0.001240,
                    0.015124,   0.007355,   0.020331,   0.011735,   0.002066,
                    0.020661,   0.041653,   0.067107,   0.015124,   0.003471,
                    0.012975,   0.056281,   0.020165,   0.027603,   0.084050,
                    0.021240,   0.052975,   0.005868,   0.008678,   0.000165,
                    0.037107,   0.017355,   0.006198,   0.011983,   0.008182,
                    0.000165,   0.003802,   0.007438,   0.013306,   0.023058,
                    0.002397,   0.002810,   0.013306,   0.001240,   0.001322,
                    0.002562,   0.046446,   0.062397,   0.067025,   0.004959,
                    0.023141,   0.005372,   0.012645,   0.039669,   0.010661,
                    0.013388,   0.000826,   0.015289,   0.012893,   0.071488,
                    0.002645,   0.002645,   0.016942,   0.016612,   0.009421,
                    0.002314,   0.002149,   0.000992,   0.000992,   0.000992,
                    0.001322,   0.013306,   0.000579,   0.000579,   0.002149,
                    0.002066,   0.011405,   0.013306,   0.004959,   0.002479,
                    0.013554,   0.000909,   0.017190,   0.004826,   0.058934,
                    0.054711,   0.058603,   0.102636,   0.024207,   0.105207,
                    0.123967,   0.050000,   0.067438,   0.066760,   0.065620,
                    0.043479,   0.039570,   0.075446,   0.082636,   0.066860,
                    0.104554,   0.046124,   0.114017,   0.078512,   0.062562,
                    0.084570,   0.075727,   0.056008,   0.090719,   0.048942,
                    0.036157,   0.108521,   0.052810,   0.056231,   0.168512,
                    0.049587,   0.165289,   0.046281,   0.055372,   0.046281,
                    0.047107,   0.001818,   0.040562,   0.009017,   0.009017,
                    0.005793,   0.022314,   0.043802,   0.006612,   0.035537,
                    0.006612,   0.029686,   0.015050,   0.016827,   0.047934,
                    0.023967,   0.027273,   0.025620,   0.004132,   0.028099,
                    0.029752,   0.028099,   0.058678,   0.044463,   0.071818,
                    0.084322,   0.046124,   0.099901,   0.012562,   0.105041,
                    0.069421,   0.015703,   0.024050,   0.064628,   0.021240,
                    0.019826,   0.015909,   0.005793,   0.017364,   0.018934,
                    0.018934,   0.002479,   0.048397,   0.036364,   0.023950,
                    0.082636,   0.069421,   0.017364,   0.033802,   0.044149,
                    0.008793,   0.048760,   0.040273,   0.084959,   0.127744,
                    0.100876,   0.142066,   0.104463,   0.103752,   0.042893,
                    0.053802,   0.051570,   0.081322,   0.068058,   0.003471,
                    0.058876,   0.009876,   0.017851,   0.141322,   0.101364,
                    0.039570,   0.012479,   0.015124,   0.014050,   0.002314,
                    0.019422,   0.016909,   0.063876,   0.012636,   0.003802,
                    0.053141,   0.068182,   0.057686,   0.055893,   0.032322,
                    0.051653,   0.029496,   0.026777,   0.017298,   0.067603,
                    0.046281,   0.034793,   0.020653,   0.075140,   0.057934,
                    0.094298,   0.083554,   0.032793,   0.034628,   0.074669,
                    0.041124,   0.068934,   0.085446,   0.035289,   0.122149,
                    0.000083,   0.060331,   0.124397,   0.010281,   0.054050,
                    0.069339,   0.042231,   0.081744,   0.077099,   0.038182,
                    0.057851,   0.033223,   0.047207,   0.023281,   0.060000,
                    0.013223,   0.118182,   0.020827,   0.025686,   0.014132,
                    0.118017,   0.049174,   0.004554,   0.008760,   0.006033,
                    0.004793,   0.021984,   0.055124,   0.021636,   0.038141,
                    0.010289,   0.017364,   0.000141,   0.012793,   0.024744,
                    0.024744,   0.023826,   0.012793,   0.019438,   0.010992,
                    0.009091,   0.049984,   0.013388,   0.017967,   0.072314,
                    0.005372,   0.005950,   0.013314,   0.035578,   0.039727,
                    0.003744,   0.073488,   0.076612,   0.095702,   0.058554,
                    0.017521,   0.067107,   0.009587,   0.092132,   0.061438,
                    0.097521,   0.032893,   0.014132,   0.096529,   0.049587,
                    0.049669,   0.079818,   0.066711,   0.002347,   0.000141,
                    0.023802,   0.015124,   0.015446,   0.052066,   0.052066,
                    0.006686,   0.018017,   0.039570,   0.029174,   0.007934,
                    0.076893,   0.003967,   0.003967,   0.004826,   0.023488,
                    0.026446,   0.081066,   0.019223,   0.014207,   0.039983,
                    0.048901,   0.012149,   0.002727,   0.058182,   0.113802,
                    0.004132,   0.037364,   0.063636,   0.029174,   0.046529,
                    0.094298,   0.054628,   0.010000,   0.005289,   0.042984,
                    0.034050,   0.062149,   0.054380,   0.047934,   0.071488,
                    0.056942,   0.041793,   0.032066,   0.094876,   0.022066,
                    0.039289,   0.085950,   0.095446,   0.059422,   0.091818,
                    0.087934,   0.049256,   0.081818,   0.103983,   0.032818,
                    0.071504,   0.054000,   0.110008,   0.004488,   0.004628,
                    0.086612,   0.073802,   0.069984,   0.003455,   0.121603,
                    0.052149,   0.037041,   0.006562,   0.012438,   0.007256,
                    0.011397,   0.055041,   0.030479,   0.010248,   0.006686,
                    0.030521,   0.102769,   0.053223,   0.000686,   0.018182,
                    0.078397,   0.018256,   0.024744,   0.005157,   0.010578,
                    0.031570,   0.091636,   0.038017,   0.074289,   0.087744,
                    0.094207,   0.080413,   0.065537,   0.114876,   0.030231,
                    0.067603,   0.025537,   0.065950,   0.071066,   0.016446,
                    0.007719,   0.057364,   0.013190,   0.034248,   0.026413,
                    0.010727,   0.020157,   0.033719,   0.058099,   0.121570,
                    0.069669,   0.018347,   0.018347,   0.058843,   0.018603,
                    0.140496,   0.027438,   0.041984,   0.021744,   0.022984,
                    0.073554,   0.079793,   0.133876,   0.098430,   0.245446,
                    0.021488,   0.069421,   0.036289,   0.059917,   0.004628,
                    0.013058,   0.075570,   0.095041,   0.101653,   0.027273,
                    0.026711,   0.025479,   0.027422,   0.042066,   0.086777,
                    0.070248,   0.103132,   0.040397,   0.081818,   0.019826,
                    0.027273,   0.042984,   0.055372,   0.041347,   0.035091,
                    0.020661,   0.059364,   0.023141,   0.080992,   0.089256,
                    0.044628,   0.041322,   0.028099,   0.034711,   0.066124,
                    0.068058,   0.045496,   0.044628,   0.016529,   0.097041,
                    0.056777,   0.045198,   0.044603,   0.097868,   0.147430,
                    0.127529,   0.035124,   0.035124,   0.077686,   0.068603,
                    0.009917,   0.009091,   0.009917,   0.010744,   0.011570,
                    0.034711,   0.028463,   0.015289,   0.022017,   0.044669,
                    0.042653,   0.078512,   0.041322,   0.091653,   0.015703,
                    0.018570,   0.005124,   0.004934,   0.013223,   0.049174,
                    0.022744,   0.024719,   0.051240,   0.027273,   0.059504,
                    0.023141,   0.030479,   0.031397,   0.040496,   0.021554,
                    0.017950,   0.033901,   0.032190,   0.001033,   0.007603,
                    0.000686,   0.007463,   0.014851,   0.027769,   0.004488,
                    0.012628,   0.004488,   0.021074,   0.004488,   0.061091,
                    0.005207,   0.006612,   0.006612,   0.023603,   0.009372,
                    0.009372,   0.000397,   0.000314,   0.018182,   0.070124,
                    0.011802,   0.020827,   0.076859,   0.088430,   0.022909,
                    0.012496,   0.017702,   0.002083,   0.003471,   0.013876,
                    0.031934,   0.021868,   0.063587,   0.041529,   0.018934,
                    0.025620,   0.012397,   0.057744,   0.007603,   0.007603,
                    0.076859,   0.008331,   0.008331,   0.009091,   0.023967,
                    0.014876,   0.014876,   0.042066,   0.027273,   0.030579,
                    0.080083,   0.031537,   0.102479,   0.009719,   0.119008,
                    0.109917,   0.056612,   0.060207,   0.025686,   0.040959,
                    0.023256,   0.009669,   0.049587,   0.031397,   0.222983,
                    0.066124,   0.055628,   0.041124,   0.102314,   0.113669,
                    0.018017,   0.021901,   0.017364,   0.037438,   0.015372,
                    0.018934,   0.058182,   0.113314,   0.129174,   0.019174,
                    0.063388,   0.088851,   0.136612,   0.039421,   0.118777,
                    0.139091,   0.050174,   0.037438,   0.046612,   0.058182,
                    0.062207,   0.069339,   0.055620,   0.084463,   0.107504,
                    0.085124,   0.018893,   0.057438,   0.091554,   0.109421,
                    0.093967,   0.007273,   0.059504,   0.067901,   0.065289,
                    0.177190,   0.129091,   0.107438,   0.068760,   0.129587,
                    0.033223,   0.130248,   0.087364,   0.038017,   0.082636,
                    0.081818,   0.059917,   0.058678,   0.159496,   0.128099,
                    0.056033,   0.068868,   0.054273,   0.000661,   0.038603,
                    0.093223,   0.058992,   0.079917,   0.091570,   0.114050,
                    0.019826,   0.026529,   0.057016,   0.079008,   0.007521,
                    0.062810,   0.010826,   0.072314,   0.056198,   0.080000,
                    0.041818,   0.030661,   0.094380,   0.010578,   0.023802,
                    0.004959,   0.004959,   0.009917,   0.026281,   0.057016,
                    0.010578,   0.007521,   0.012984,   0.014207,   0.113388,
                    0.113058,   0.033058,   0.093388,   0.121653,   0.053719,
                    0.020744,   0.096686,   0.006281,   0.021157,   0.023802,
                    0.104132,   0.078512,   0.007603,   0.033802,   0.025612,
                    0.089942,   0.044628,   0.028099,   0.052793,   0.109950,
                    0.018934,   0.018934,   0.007190,   0.003058,   0.013719,
                    0.068760,   0.120661,   0.062364,   0.016364,   0.005372,
                    0.145446,   0.078760,   0.060000,   0.017934,   0.120661,
                    0.009917,   0.119008,   0.068099,   0.104868,   0.051397,
                    0.079678,   0.032388,   0.072314,   0.017769,   0.027917,
                    0.021579,   0.054554,   0.021744,   0.007017,   0.089504,
                    0.069421,   0.041322,   0.057364,   0.023802,   0.010744,
                    0.078934,   0.008256,   0.004959,   0.092562,   0.035537,
                    0.013554,   0.010744,   0.016529,   0.102810,   0.068603,
                    0.098843,   0.026860,   0.057016,   0.057769,   0.040174,
                    0.107438,   0.016529,   0.007438,   0.007438,   0.007438,
                    0.007438,   0.007438,   0.082157,   0.052893,   0.068430,
                    0.034876,   0.052231,   0.024554,   0.023141,   0.076678,
                    0.098959,   0.191901,   0.081066,   0.110744,   0.033058,
                    0.071066,   0.071066,   0.008256,   0.013719,   0.029686,
                    0.027769,   0.078256,   0.099826,   0.062149,   0.064711,
                    0.071322,   0.060992,   0.071322,   0.065950,   0.047471,
                    0.015703,   0.017364,   0.097521,   0.034711,   0.028934,
                    0.004132,   0.004132,   0.004132,   0.034711,   0.004132,
                    0.004132,   0.004132,   0.118182,   0.056860,   0.108099,
                    0.180174,   0.025950,   0.036364,   0.090083,   0.047141,
                    0.016504,   0.000661,   0.073223,   0.074380,   0.003471,
                    0.010744,   0.014876,   0.009339,   0.012727,   0.015289,
                    0.015289,   0.043802,   0.019826,   0.111901,   0.022397,
                    0.021488,   0.021488,   0.052446,   0.031760,   0.016859,
                    0.010909,   0.024793,   0.023141,   0.009917,   0.020661,
                    0.004132,   0.004132,   0.067273,   0.017364,   0.015703,
                    0.004207,   0.022314,   0.028934,   0.023719,   0.010744,
                    0.006942,   0.036364,   0.009256,   0.007438,   0.023016,
                    0.023016,   0.004959,   0.124554,   0.069587,   0.093554,
                    0.106504,   0.103058,   0.079025,   0.069537,   0.004132,
                    0.014876,   0.014876,   0.021488,   0.043967,   0.125620,
                    0.037686,   0.069174,   0.009174,   0.182727,   0.017521,
                    0.050826,   0.128934,   0.074959,   0.069421,   0.061157,
                    0.121488,   0.062810,   0.115702,   0.097521,   0.017273,
                    0.093471,   0.013719,   0.017364,   0.075868,   0.037686,
                    0.111066,   0.091157,   0.043471,   0.013719,   0.013719,
                    0.068760,   0.094207,   0.042984,   0.042984,   0.055537,
                    0.062149,   0.049504,   0.071570,   0.057934,   0.030083,
                    0.093388,   0.004207,   0.029752,   0.164298,   0.064132,
                    0.051240,   0.023967,   0.116686,   0.146281,   0.089422,
                    0.065289,   0.014132,   0.061901,   0.003876,   0.012636,
                    0.091744,   0.086124,   0.010578,   0.095868,   0.016529,
                    0.065289,   0.031397,   0.065289,   0.031397,   0.074000,
                    0.005289,   0.028099,   0.013314,   0.066281,   0.092231,
                    0.130413,   0.066124,   0.084711,   0.084711,   0.033554,
                    0.030496,   0.008264,   0.010000,   0.000909,   0.018264,
                    0.000207,   0.000207,   0.000063,   0.000207,   0.000207,
                    0.000063,   0.000207,   0.000207,   0.000207,   0.000063,
                    0.000207,   0.000207,   0.000207,   0.000063,   0.000063,
                    0.000207,   0.000063,   0.000207,   0.000063,   0.000207,
                    0.000207,   0.000063,   0.000207,   0.000207,   0.000207,
                    0.000207,   0.000207,   0.000207,   0.000207,   0.000207,
                    0.000207,   0.000063,   0.000207,   0.000207,   0.000207,
                    0.000207,   0.000207,   0.000207,   0.000207,   0.000207,
                    0.000207,   0.000207,   0.000207,   0.000207,   0.000063,
                    0.000063,   0.000207,   0.000063,   0.000207,   0.000207,
                    0.000063,   0.000207,   0.000207,   0.000207,   0.000207,
                    0.000063,   0.000207,   0.000207,   0.000063,   0.000207,
                    0.000063,   0.000207,   0.000207,   0.000207,   0.000063,
                    0.000207,   0.000207,   0.000207,   0.000063,   0.000063,
                    0.000207,   0.000207,   0.000207,   0.000063,   0.000207,
                    0.000063,   0.000207,   0.000207,   0.000063,   0.000207,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000100,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000100,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000100,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000100,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000100,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000100,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000100,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000100,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000100,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000100,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826,   0.000826,
                    0.000826,   0.000826,   0.000826,   0.000826])
    
    G_br = np.zeros(len(X_br))
 
    B_br = np.array([0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,  -0.008320,  -0.008000,  -0.016000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.099200,   0.099200,   0.126034,   0.117418,   0.116838,
                    0.505760,   0.046948,   0.046948,   0.643200,   0.643200,
                    0.898240,   0.141134,   0.111417,   0.096994,   0.196988,
                    0.029621,   0.094961,   0.170852,   0.114805,   0.103673,
                    0.170775,   0.102802,   0.129935,   0.134842,   0.148685,
                    0.130196,   0.159720,   0.100962,   0.515840,   0.403200,
                    0.921600,   0.506240,   0.411840,   0.701440,   0.664000,
                    0.451200,   1.404800,   0.858880,   0.595840,   0.144232,
                    0.060210,   0.049852,   0.165528,   0.140166,   0.088959,
                    0.245194,   0.075533,   0.075533,   0.061952,   0.000000,
                    0.011132,   0.022429,   0.049620,   0.024781,   0.018198,
                    0.018198,   0.045883,   0.935040,   1.094400,   1.021630,
                    0.918400,   0.341248,   0.228480,   0.092047,   0.100662,
                    0.098349,   0.105802,   0.066986,   0.066986,   0.073481,
                    0.031460,   0.079666,   0.042689,   0.087894,   0.063598,
                    0.051691,   0.013165,   0.065727,   0.069309,   0.093712,
                    0.027588,   0.022525,   0.541760,   0.341760,   0.707840,
                    0.293760,   0.374880,   0.573440,   0.059609,   0.084390,
                    0.022845,   0.052640,   0.007579,   0.060055,   0.008683,
                    0.116818,   0.025749,   0.163611,   0.072145,   0.044983,
                    0.061197,   0.061197,   0.058748,   0.022603,   0.046115,
                    0.115705,   0.073636,   0.020551,   0.016243,   0.007008,
                    0.009544,   0.009477,   0.019128,   0.038391,   0.055312,
                    0.017598,   0.046154,   0.063907,   0.048671,   0.052349,
                    0.036165,   0.022399,   0.014239,   0.039969,   0.022390,
                    0.066172,   0.036165,   0.022109,   0.007260,   0.007241,
                    0.767968,   0.677440,   0.289888,   0.991072,   0.292928,
                    0.126614,   0.126614,   0.140070,   0.140070,   0.069793,
                    0.124127,   0.070664,   0.159139,   0.088475,   0.145394,
                    0.063598,   0.122065,   0.086733,   0.025410,   0.215196,
                    0.021296,   0.041237,   0.116644,   0.087701,   0.052456,
                    0.050723,   0.005711,   0.005711,   0.122355,   0.053821,
                    0.069212,   0.010280,   0.070954,   0.024781,   0.058951,
                    0.027588,   0.705920,   0.456320,   0.433664,   1.186880,
                    0.253810,   0.242871,   0.167851,   0.090798,   0.051498,
                    0.083635,   0.064372,   0.129712,   0.411520,   0.723520,
                    0.657920,   0.469440,   0.418208,   0.418208,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,  -0.021440,  -0.021440,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,  -0.015040,  -0.015040,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,  -0.003001,
                    -0.003194,  -0.005324,  -0.009920,  -0.009920,  -0.003098,
                    -0.003194,  -0.002517,  -0.002517,  -0.002323,  -0.002807,
                    -0.006582,  -0.002904,  -0.003194,  -0.002323,  -0.003194,
                    -0.002226,  -0.003194,  -0.002904,  -0.002904,  -0.003388,
                    -0.003001,  -0.002226,  -0.005227,  -0.003485,  -0.002904,
                    -0.008640,  -0.008640,  -0.002226,  -0.006582,  -0.015040,
                    -0.009920,  -0.009920,  -0.003001,  -0.002323,  -0.002904,
                    -0.003775,  -0.002904,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.004162,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.004162,
                    0.000000,  -0.019840,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    -0.019840,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.009680,   0.006241,
                    0.006411,   0.011069,   0.010140,   0.010481,   0.010070,
                    0.007241,   0.007860,   0.012439,   0.008661,   0.008521,
                    0.006561,   0.012410,   0.008809,   0.004709,   0.007650,
                    0.012901,   0.005111,   0.017981,   0.005711,   0.004859,
                    0.010159,   0.022170,   0.012049,   0.012199,   0.011081,
                    0.018760,   0.006950,   0.007841,   0.031511,   0.007190,
                    0.012320,   0.006190,   0.013000,   0.003560,   0.009169,
                    0.004620,   0.010890,   0.012359,   0.013000,   0.006389,
                    0.005689,   0.005130,   0.007991,   0.006970,   0.011110,
                    0.007059,   0.010919,   0.010159,   0.019079,   0.009361,
                    0.007410,   0.006820,   0.010210,   0.006921,   0.004361,
                    0.002950,   0.006590,   0.002321,   0.001839,   0.000999,
                    0.006290,   0.006290,   0.004339,   0.005871,   0.005590,
                    0.005491,   0.000871,   0.006781,   0.004850,   0.007541,
                    0.005810,   0.008180,   0.010650,   0.019089,   0.010341,
                    0.006520,   0.006941,   0.007889,   0.000820,   0.007500,
                    0.007589,   0.009920,   0.004610,   0.007950,   0.010450,
                    0.007889,   0.008049,   0.002229,   0.012100,   0.007500,
                    0.009690,   0.009951,   0.015730,   0.007889,   0.002420,
                    0.004600,   0.005801,   0.003291,   0.002069,   0.001539,
                    0.002210,   0.002096,   0.001191,   0.002021,   0.002021,
                    0.001951,   0.002081,   0.002887,   0.003751,   0.003001,
                    0.001050,   0.002708,   0.000530,   0.000709,   0.005220,
                    0.004320,   0.001200,   0.001200,   0.001960,   0.000351,
                    0.000699,   0.004840,   0.001660,   0.003640,   0.002870,
                    0.000540,   0.002190,   0.003190,   0.008661,   0.009269,
                    0.000361,   0.004291,   0.001498,   0.003531,   0.000150,
                    0.001261,   0.031039,   0.001781,   0.001350,   0.009929,
                    0.000881,   0.001019,   0.000179,   0.001191,   0.002739,
                    0.002180,   0.001980,   0.000699,   0.004221,   0.001191,
                    0.000150,   0.000731,   0.000820,   0.001060,   0.003161,
                    0.004099,   0.005779,   0.000891,   0.002410,   0.001261,
                    0.006490,   0.004361,   0.002931,   0.006979,   0.019491,
                    0.001401,   0.012100,   0.012100,   0.001169,   0.001370,
                    0.011139,   0.007040,   0.011490,   0.006950,   0.003630,
                    0.002969,   0.009670,   0.002979,   0.002069,   0.003071,
                    0.002621,   0.002120,   0.003720,   0.012920,   0.001101,
                    0.007850,   0.001450,   0.019740,   0.000961,   0.001319,
                    0.005421,   0.001839,   0.003151,   0.010980,   0.001319,
                    0.001290,   0.013230,   0.006290,   0.013741,   0.013230,
                    0.006749,   0.010999,   0.001101,   0.001169,   0.006490,
                    0.001989,   0.005210,   0.003901,   0.003020,   0.007739,
                    0.003920,   0.002340,   0.001181,   0.002139,   0.002299,
                    0.006551,   0.008429,   0.001769,   0.005750,   0.008901,
                    0.007432,   0.004600,   0.001830,   0.001909,   0.003521,
                    0.001931,   0.000731,   0.000999,   0.001791,   0.004700,
                    0.001450,   0.002420,   0.000329,   0.002991,   0.001670,
                    0.001251,   0.000210,   0.001041,   0.001890,   0.002481,
                    0.000779,   0.001900,   0.001210,   0.002161,   0.001341,
                    0.002701,   0.003959,   0.005051,   0.001699,   0.003161,
                    0.000900,   0.000041,   0.001111,   0.000820,   0.003831,
                    0.005680,   0.001549,   0.000450,   0.005711,   0.005089,
                    0.001229,   0.000450,   0.000840,   0.002611,   0.000801,
                    0.001750,   0.002669,   0.003151,   0.004339,   0.002081,
                    0.004061,   0.001909,   0.008550,   0.001319,   0.001769,
                    0.001810,   0.008840,   0.008410,   0.000489,   0.002180,
                    0.003531,   0.002461,   0.003969,   0.004700,   0.003417,
                    0.005501,   0.003461,   0.008250,   0.001200,   0.004441,
                    0.008080,   0.002669,   0.007470,   0.001389,   0.008601,
                    0.013390,   0.000871,   0.004000,   0.001191,   0.001140,
                    0.002168,   0.009140,   0.010060,   0.000789,   0.004020,
                    0.003589,   0.003301,   0.004361,   0.000961,   0.002710,
                    0.004610,   0.001341,   0.006599,   0.001210,   0.009619,
                    0.002289,   0.007500,   0.004929,   0.006539,   0.006909,
                    0.002851,   0.001641,   0.002790,   0.013741,   0.008151,
                    0.001421,   0.001341,   0.005885,   0.000944,   0.001989,
                    0.000000,   0.006539,   0.011841,   0.008760,   0.019319,
                    0.002180,   0.004051,   0.009760,   0.005820,   0.006650,
                    0.008027,   0.007350,   0.002921,   0.002490,   0.003049,
                    0.005271,   0.006670,   0.001861,   0.005329,   0.003611,
                    0.009051,   0.006340,   0.006561,   0.008310,   0.002720,
                    0.007850,   0.012879,   0.006079,   0.010060,   0.003550,
                    0.006229,   0.006989,   0.000690,   0.001801,   0.001031,
                    0.003199,   0.003579,   0.001571,   0.016570,   0.010181,
                    0.001561,   0.003030,   0.002241,   0.004291,   0.003691,
                    0.004431,   0.014469,   0.007720,   0.007059,   0.015851,
                    0.003071,   0.006839,   0.011991,   0.000990,   0.002311,
                    0.012981,   0.016650,   0.008891,   0.010500,   0.010989,
                    0.002401,   0.009699,   0.003681,   0.009731,   0.008981,
                    0.002841,   0.014460,   0.001379,   0.004160,   0.006870,
                    0.000881,   0.006561,   0.009169,   0.004010,   0.011480,
                    0.006820,   0.014890,   0.005699,   0.010159,   0.009699,
                    0.007350,   0.006440,   0.011040,   0.011021,   0.001169,
                    0.018329,   0.011251,   0.007739,   0.003860,   0.006730,
                    0.009699,   0.001440,   0.014130,   0.002890,   0.007739,
                    0.006561,   0.007831,   0.007350,   0.007066,   0.010060,
                    0.008511,   0.000755,   0.235708,   0.218478,   0.235660,
                    0.002568,   0.000024,   0.009740,   0.010140,   0.008850,
                    0.007461,   0.014230,   0.013211,   0.010699,   0.002759,
                    0.005421,   0.013591,   0.008180,   0.008180,   0.006701,
                    0.013211,   0.010580,   0.012959,   0.012751,   0.020321,
                    0.000670,   0.008090,   0.004020,   0.006241,   0.002909,
                    0.003410,   0.006829,   0.004709,   0.000920,   0.000750,
                    0.000849,   0.002081,   0.001951,   0.008470,   0.009699,
                    0.004891,   0.000779,   0.000990,   0.006241,   0.008301,
                    0.001360,   0.006781,   0.002570,   0.001200,   0.001801,
                    0.000750,   0.000440,   0.001050,   0.010999,   0.004521,
                    0.002570,   0.001609,   0.004179,   0.006510,   0.009341,
                    0.007231,   0.010600,   0.001890,   0.010989,   0.021081,
                    0.004540,   0.002730,   0.003729,   0.002931,   0.001331,
                    0.010680,   0.013361,   0.001309,   0.004310,   0.010041,
                    0.012579,   0.003180,   0.000900,   0.008710,   0.006369,
                    0.010140,   0.010290,   0.006340,   0.008320,   0.001730,
                    0.003659,   0.006711,   0.001379,   0.000179,   0.005150,
                    0.001229,   0.005150,   0.005280,   0.013141,   0.004230,
                    0.009540,   0.003359,   0.000939,   0.002219,   0.004751,
                    0.010699,   0.014750,   0.011510,   0.002979,   0.003260,
                    0.007381,   0.010411,   0.012470,   0.009341,   0.006921,
                    0.001871,   0.002851,   0.003119,   0.001801,   0.009179,
                    0.003845,   0.007865,   0.002771,   0.006079,   0.006839,
                    0.016129,   0.003381,   0.008799,   0.005810,   0.012419,
                    0.019031,   0.005501,   0.002979,   0.000549,   0.001801,
                    0.002771,   0.002410,   0.013961,   0.002570,   0.008729,
                    0.010341,   0.011541,   0.000680,   0.003509,   0.005699,
                    0.001980,   0.011580,   0.006401,   0.001980,   0.006790,
                    0.007289,   0.001520,   0.000661,   0.005351,   0.003020,
                    0.001730,   0.001660,   0.001210,   0.007480,   0.022351,
                    0.002040,   0.001600,   0.002471,   0.001089,   0.008981,
                    0.003221,   0.007570,   0.004400,   0.001801,   0.007429,
                    0.007410,   0.006711,   0.012630,   0.008061,   0.002321,
                    0.007819,   0.007381,   0.006110,   0.006950,   0.003770,
                    0.006861,   0.002631,   0.002040,   0.005230,   0.000581,
                    0.003390,   0.009240,   0.001430,   0.010941,   0.010909,
                    0.006829,   0.004881,   0.001229,   0.001370,   0.000830,
                    0.001730,   0.012879,   0.004620,   0.006490,   0.003250,
                    0.008071,   0.005871,   0.002710,   0.006749,   0.002810,
                    0.002101,   0.005929,   0.000249,   0.000191,   0.009489,
                    0.005191,   0.007050,   0.002710,   0.003011,   0.002040,
                    0.005990,   0.001650,   0.005159,   0.007640,   0.000051,
                    0.005670,   0.004496,   0.013119,   0.002449,   0.001549,
                    0.009000,   0.006140,   0.004540,   0.005060,   0.005610,
                    0.001050,   0.005871,   0.006861,   0.002979,   0.007991,
                    0.007700,   0.001500,   0.002420,   0.010641,   0.011471,
                    0.008010,   0.000801,   0.006759,   0.012630,   0.009530,
                    0.001370,   0.010079,   0.005431,   0.001750,   0.003291,
                    0.010379,   0.000820,   0.000820,   0.024200,   0.005411,
                    0.011580,   0.002510,   0.004881,   0.001699,   0.005159,
                    0.004051,   0.009489,   0.007279,   0.001009,   0.001481,
                    0.004649,   0.002519,   0.001430,   0.001890,   0.007020,
                    0.002691,   0.003959,   0.001960,   0.003221,   0.008620,
                    0.004501,   0.004961,   0.001890,   0.008030,   0.001650,
                    0.002810,   0.005421,   0.000970,   0.001270,   0.001331,
                    0.001191,   0.005890,   0.005890,   0.003589,   0.005680,
                    0.005680,   0.006880,   0.000290,   0.006331,   0.008240,
                    0.005019,   0.001220,   0.000920,   0.007579,   0.006411,
                    0.001500,   0.006140,   0.006941,   0.003589,   0.006103,
                    0.009048,   0.001004,   0.008291,   0.000010,   0.005840,
                    0.000044,   0.005893,   0.000172,   0.011081,   0.001832,
                    0.010571,   0.000019,   0.004131,   0.008361,   0.000135,
                    0.004136,   0.008361,   0.000135,   0.013225,   0.011890,
                    0.004112,   0.004112,   0.000886,   0.002664,   0.000082,
                    0.004898,   0.007047,   0.003231,   0.005145,   0.002331,
                    0.001488,   0.002079,   0.002183,   0.000549,   0.001774,
                    0.003628,   0.000908,   0.001774,   0.003628,   0.000900,
                    0.007071,   0.006333,   0.005322,   0.005084,   0.002183,
                    0.000879,   0.002217,   0.001793,   0.000879,   0.005370,
                    0.005370,   0.000983,   0.001878,   0.001980,   0.000554,
                    0.001919,   0.010346,   0.001421,   0.005704,   0.007771,
                    0.000864,   0.000852,   0.003782,   0.004259,   0.003364,
                    0.003015,   0.010285,   0.012320,   0.003040,   0.007168,
                    0.004753,   0.004973,   0.005878,   0.007795,   0.007366,
                    0.007124,   0.002798,   0.005242,   0.014394,   0.009402,
                    0.005890,   0.002848,   0.006740,   0.000370,   0.004864,
                    0.001861,   0.006556,   0.012681,   0.006437,   0.005135,
                    0.002343,   0.005234,   0.002331,   0.002335,   0.025112,
                    0.023334,   0.011374,   0.000077,   0.000692,   0.000721,
                    0.000194,   0.000029,   0.006278,   0.006486,   0.006396,
                    0.011805,   0.010803,   0.000997,   0.004559,   0.002756,
                    0.007577,   0.000007,   0.001128,   0.000024,   0.000443,
                    0.000707,   0.000276,   0.002998,   0.000068,   0.000208,
                    0.000833,   0.000833,   0.001735,   0.001191,   0.001191,
                    0.006691,   0.002979,   0.001496,   0.000165,   0.001481,
                    0.001549,   0.003151,   0.006863,   0.001757,   0.003884,
                    0.003090,   0.009854,   0.003095,   0.005665,   0.002841,
                    0.001447,   0.004264,   0.003178,   0.007267,   0.010846,
                    0.012221,   0.007921,   0.008523,   0.000007,   0.003308,
                    0.003073,   0.001493,   0.011437,   0.003110,   0.001125,
                    0.000010,   0.000010,   0.002464,   0.000632,   0.001847,
                    0.001847,   0.006548,   0.001815,   0.002563,   0.007163,
                    0.004997,   0.004257,   0.000806,   0.006350,   0.000494,
                    0.003403,   0.002943,   0.001619,   0.000903,   0.001689,
                    0.000992,   0.004371,   0.006200,   0.001484,   0.003035,
                    0.000975,   0.003596,   0.006086,   0.001747,   0.006510,
                    0.004291,   0.001585,   0.006110,   0.009864,   0.002331,
                    0.009039,   0.004867,   0.010016,   0.000949,   0.005114,
                    0.000980,   0.008327,   0.002594,   0.002594,   0.006515,
                    0.002154,   0.001290,   0.006902,   0.001810,   0.005283,
                    0.004378,   0.000997,   0.007446,   0.003969,   0.003030,
                    0.001658,   0.005385,   0.004646,   0.000024,   0.001670,
                    0.006735,   0.003599,   0.001767,   0.000169,   0.000322,
                    0.000044,   0.000779,   0.004371,   0.002819,   0.002072,
                    0.000036,   0.000121,   0.001619,   0.004898,   0.009043,
                    0.002389,   0.001283,   0.002118,   0.003707,   0.004825,
                    0.001566,   0.000237,   0.005467,   0.000269,   0.000779,
                    0.002222,   0.000007,   0.000704,   0.003025,   0.001675,
                    0.000404,   0.000266,   0.002684,   0.000075,   0.002553,
                    0.002633,   0.005193,   0.003453,   0.001251,   0.000789,
                    0.003233,   0.000346,   0.001951,   0.000281,   0.004262,
                    0.001433,   0.001873,   0.001774,   0.000646,   0.000017,
                    0.000426,   0.000549,   0.002297,   0.002323,   0.000162,
                    0.000339,   0.000617,   0.004484,   0.001275,   0.002268,
                    0.001517,   0.000254,   0.000404,   0.001515,   0.001101,
                    0.000469,   0.000857,   0.001370,   0.006459,   0.000349,
                    0.000530,   0.001474,   0.000462,   0.000007,   0.000973,
                    0.000728,   0.000017,   0.001125,   0.000010,   0.000387,
                    0.001614,   0.000281,   0.000387,   0.001375,   0.000225,
                    0.002592,   0.001534,   0.000970,   0.004104,   0.000019,
                    0.000053,   0.000177,   0.000177,   0.004414,   0.000390,
                    0.000128,   0.003112,   0.001396,   0.000293,   0.000753,
                    0.000160,   0.000511,   0.001500,   0.000252,   0.004211,
                    0.001619,   0.002043,   0.001646,   0.000237,   0.000237,
                    0.000779,   0.000779,   0.000999,   0.001626,   0.001290,
                    0.002628,   0.000566,   0.002660,   0.000351,   0.000293,
                    0.000046,   0.003124,   0.002715,   0.003277,   0.000777,
                    0.002715,   0.002386,   0.000692,   0.000857,   0.001379,
                    0.000721,   0.000740,   0.000378,   0.000665,   0.000194,
                    0.002413,   0.001718,   0.001430,   0.000697,   0.001752,
                    0.000194,   0.000811,   0.001994,   0.000019,   0.000895,
                    0.000489,   0.002439,   0.000191,   0.002413,   0.000116,
                    0.000847,   0.002725,   0.001242,   0.000542,   0.000728,
                    0.000617,   0.001919,   0.001752,   0.000194,   0.001805,
                    0.004726,   0.002618,   0.001752,   0.003030,   0.003913,
                    0.000816,   0.001726,   0.000578,   0.001914,   0.001217,
                    0.000593,   0.000220,   0.001004,   0.002030,   0.000927,
                    0.000467,   0.001805,   0.000107,   0.002338,   0.002645,
                    0.001788,   0.008259,   0.000879,   0.001876,   0.000748,
                    0.000046,   0.002517,   0.004075,   0.000387,   0.005365,
                    0.002698,   0.000218,   0.000726,   0.000956,   0.001104,
                    0.001796,   0.000927,   0.006558,   0.003533,   0.000525,
                    0.001667,   0.001561,   0.002522,   0.000019,   0.000240,
                    0.002294,   0.000298,   0.002664,   0.002778,   0.004136,
                    0.005537,   0.003243,   0.000772,   0.000010,   0.001711,
                    0.001636,   0.009319,   0.002829,   0.002616,   0.000423,
                    0.002139,   0.004477,   0.005953,   0.001779,   0.002130,
                    0.001314,   0.003683,   0.003265,   0.000472,   0.002773,
                    0.002863,   0.000542,   0.002161,   0.002139,   0.000271,
                    0.000281,   0.000702,   0.001239,   0.002335,   0.001706,
                    0.003083,   0.001801,   0.000963,   0.004455,   0.003013,
                    0.001159,   0.005189,   0.003040,   0.003165,   0.003202,
                    0.003006,   0.003006,   0.001559,   0.001556,   0.001375,
                    0.003112,   0.001544,   0.004499,   0.002120,   0.001638,
                    0.001440,   0.001198,   0.001619,   0.001699,   0.001261,
                    0.000520,   0.000883,   0.002142,   0.000784,   0.000828,
                    0.000823,   0.001363,   0.002139,   0.000022,   0.002091,
                    0.000857,   0.000927,   0.000133,   0.000944,   0.000590,
                    0.001159,   0.001058,   0.000012,   0.000503,   0.003836,
                    0.002807,   0.005377,   0.003705,   0.000564,   0.002176,
                    0.001171,   0.002243,   0.000876,   0.003030,   0.002904,
                    0.001696,   0.008981,   0.001195,   0.000513,   0.000329,
                    0.002410,   0.003040,   0.000530,   0.000726,   0.001440,
                    0.000232,   0.000051,   0.001592,   0.000257,   0.000820,
                    0.000358,   0.001907,   0.000152,   0.000145,   0.000145,
                    0.000128,   0.001769,   0.002134,   0.002802,   0.002176,
                    0.003178,   0.000506,   0.001801,   0.001210,   0.000956,
                    0.000436,   0.002703,   0.003272,   0.001740,   0.000032,
                    0.000748,   0.012678,   0.005210,   0.017475,   0.000353,
                    0.002735,   0.006297,   0.006817,   0.007749,   0.005198,
                    0.017877,   0.004627,   0.006638,   0.000089,   0.006486,
                    0.004441,   0.002948,   0.003572,   0.002522,   0.002534,
                    0.002248,   0.005588,   0.009019,   0.000242,   0.000242,
                    0.006752,   0.006570,   0.000581,   0.006660,   0.006752,
                    0.000956,   0.000581,   0.007214,   0.007802,   0.006793,
                    0.006793,   0.006614,   0.000380,   0.000370,   0.005474,
                    0.000799,   0.003816,   0.002362,   0.008901,   0.007335,
                    0.007550,   0.001423,   0.011265,   0.008956,   0.006924,
                    0.004804,   0.007647,   0.004518,   0.002180,   0.004576,
                    0.001614,   0.000758,   0.000753,   0.006142,   0.005917,
                    0.006210,   0.005220,   0.010188,   0.003903,   0.012030,
                    0.007149,   0.010721,   0.006795,   0.005092,   0.003698,
                    0.003586,   0.001210,   0.002502,   0.011626,   0.011628,
                    0.006638,   0.002524,   0.005936,   0.014297,   0.008707,
                    0.008833,   0.004617,   0.000000,   0.000000,   0.000000,
                    0.006457,   0.001123,   0.009932,   0.000102,   0.006072,
                    0.006500,   0.008877,   0.003778,   0.009111,   0.001123,
                    0.001672,   0.000121,   0.005515,   0.000315,   0.000370,
                    0.000653,   0.000673,   0.001563,   0.002800,   0.002529,
                    0.000605,   0.000590,   0.000590,   0.000414,   0.000426,
                    0.000716,   0.001108,   0.004813,   0.001994,   0.003872,
                    0.001268,   0.000748,   0.003741,   0.004424,   0.001317,
                    0.000586,   0.000755,   0.005162,   0.003134,   0.000908,
                    0.007214,   0.003078,   0.001149,   0.016066,   0.009535,
                    0.018963,   0.011205,   0.004015,   0.009740,   0.001779,
                    0.011696,   0.006389,   0.001309,   0.007180,   0.001026,
                    0.001256,   0.001244,   0.006878,   0.002357,   0.003603,
                    0.013649,   0.000961,   0.003528,   0.004581,   0.001670,
                    0.002246,   0.001951,   0.001951,   0.004104,   0.003146,
                    0.007979,   0.002452,   0.004366,   0.001605,   0.003691,
                    0.011251,   0.001239,   0.002214,   0.007439,   0.024389,
                    0.024841,   0.000312,   0.016020,   0.012928,   0.001326,
                    0.000000,   0.000000,   0.001391,   0.006483,   0.004874,
                    0.002253,   0.002294,   0.005397,   0.000452,   0.011948,
                    0.011026,   0.016129,   0.004228,   0.000900,   0.003349,
                    0.001868,   0.000370,   0.000353,   0.000111,   0.000252,
                    0.003557,   0.005498,   0.008245,   0.004884,   0.006326,
                    0.000087,   0.001938,   0.001747,   0.000186,   0.001909,
                    0.007265,   0.000794,   0.000649,   0.000220,   0.001849,
                    0.001314,   0.001953,   0.001783,   0.005065,   0.002222,
                    0.001317,   0.002870,   0.003359,   0.002120,   0.008835,
                    0.000472,   0.005000,   0.000745,   0.004090,   0.001953,
                    0.009431,   0.001130,   0.011715,   0.005055,   0.006621,
                    0.001258,   0.000668,   0.000665,   0.003228,   0.000559,
                    0.002885,   0.003734,   0.000941,   0.003620,   0.003463,
                    0.000632,   0.001205,   0.000322,   0.001159,   0.001154,
                    0.003214,   0.002517,   0.000351,   0.000436,   0.000295,
                    0.000651,   0.001079,   0.001309,   0.000750,   0.004576,
                    0.004056,   0.001597,   0.000000,   0.001096,   0.000061,
                    0.001854,   0.000673,   0.003979,   0.002033,   0.000097,
                    0.000133,   0.000794,   0.001486,   0.000123,   0.000452,
                    0.001089,   0.000784,   0.001871,   0.000835,   0.000242,
                    0.001793,   0.000784,   0.002188,   0.001341,   0.000237,
                    0.002234,   0.003976,   0.007543,   0.001769,   0.000506,
                    0.001440,   0.005973,   0.002188,   0.002909,   0.008869,
                    0.002072,   0.005779,   0.000690,   0.000895,   0.000017,
                    0.004029,   0.001943,   0.000651,   0.001283,   0.000946,
                    0.000017,   0.000445,   0.000859,   0.001411,   0.002403,
                    0.000264,   0.000319,   0.001379,   0.000131,   0.000135,
                    0.000300,   0.004487,   0.006585,   0.007069,   0.000530,
                    0.002713,   0.000634,   0.001498,   0.004279,   0.001246,
                    0.001203,   0.000007,   0.001605,   0.001735,   0.007393,
                    0.000305,   0.000310,   0.001960,   0.001842,   0.001118,
                    0.000254,   0.000237,   0.000121,   0.000119,   0.000114,
                    0.000157,   0.001496,   0.000065,   0.000068,   0.000215,
                    0.000210,   0.001183,   0.001396,   0.000469,   0.000286,
                    0.001541,   0.000109,   0.001917,   0.000469,   0.005779,
                    0.005031,   0.005571,   0.013441,   0.002299,   0.010210,
                    0.011969,   0.005009,   0.006631,   0.006609,   0.006609,
                    0.004150,   0.003780,   0.007470,   0.008470,   0.006921,
                    0.009951,   0.004450,   0.011289,   0.008199,   0.006050,
                    0.008371,   0.007497,   0.005539,   0.008981,   0.004840,
                    0.003451,   0.010740,   0.005411,   0.005571,   0.018970,
                    0.006050,   0.018220,   0.005009,   0.005951,   0.005009,
                    0.005300,   0.026831,   0.004010,   0.000891,   0.000891,
                    0.000610,   0.002420,   0.004550,   0.000731,   0.003901,
                    0.000731,   0.002940,   0.001491,   0.001670,   0.005150,
                    0.002570,   0.002640,   0.002781,   0.000409,   0.003071,
                    0.003190,   0.003071,   0.005740,   0.004390,   0.007020,
                    0.008349,   0.004569,   0.009891,   0.001300,   0.010580,
                    0.006660,   0.000801,   0.002490,   0.006679,   0.002210,
                    0.001941,   0.001529,   0.000549,   0.001650,   0.002011,
                    0.001909,   0.000271,   0.004760,   0.003991,   0.002369,
                    0.008540,   0.007110,   0.001721,   0.003291,   0.004293,
                    0.000840,   0.004770,   0.003841,   0.006350,   0.012649,
                    0.009990,   0.013479,   0.010430,   0.009910,   0.004281,
                    0.005571,   0.005280,   0.008339,   0.006493,   0.000344,
                    0.005827,   0.000980,   0.001830,   0.013939,   0.010031,
                    0.003920,   0.001159,   0.001529,   0.001379,   0.000240,
                    0.002011,   0.001469,   0.005980,   0.001191,   0.000380,
                    0.005230,   0.006740,   0.005670,   0.005329,   0.003081,
                    0.004891,   0.002880,   0.002640,   0.001660,   0.006510,
                    0.004751,   0.003170,   0.002040,   0.007420,   0.004620,
                    0.008981,   0.007860,   0.003250,   0.003560,   0.007391,
                    0.004070,   0.006921,   0.008741,   0.003410,   0.011790,
                    0.000000,   0.006050,   0.012940,   0.001430,   0.005319,
                    0.006631,   0.004141,   0.008010,   0.007630,   0.003780,
                    0.005929,   0.003279,   0.004671,   0.002299,   0.005740,
                    0.001300,   0.011589,   0.002062,   0.002543,   0.001430,
                    0.011260,   0.004869,   0.000460,   0.000871,   0.000610,
                    0.000479,   0.002280,   0.005711,   0.002101,   0.003780,
                    0.001019,   0.001740,   0.000010,   0.001220,   0.002510,
                    0.002510,   0.002270,   0.000980,   0.001919,   0.001140,
                    0.000939,   0.004949,   0.001379,   0.001711,   0.007279,
                    0.000549,   0.000629,   0.001379,   0.003400,   0.003790,
                    0.000380,   0.007451,   0.007771,   0.009470,   0.005801,
                    0.000779,   0.006210,   0.000801,   0.009121,   0.006009,
                    0.009990,   0.003470,   0.001450,   0.009891,   0.005080,
                    0.005179,   0.007831,   0.006360,   0.000230,   0.000010,
                    0.002270,   0.001450,   0.001810,   0.004980,   0.004980,
                    0.000629,   0.001791,   0.003918,   0.002829,   0.000770,
                    0.007720,   0.000559,   0.000559,   0.000460,   0.002241,
                    0.002691,   0.008279,   0.001941,   0.001500,   0.004010,
                    0.004901,   0.001229,   0.000259,   0.006031,   0.010960,
                    0.000390,   0.003821,   0.007241,   0.002781,   0.004811,
                    0.009849,   0.005520,   0.001019,   0.000540,   0.004431,
                    0.003480,   0.006360,   0.005641,   0.004910,   0.007470,
                    0.005750,   0.004240,   0.003291,   0.009721,   0.002161,
                    0.003940,   0.008809,   0.009390,   0.005670,   0.009070,
                    0.009099,   0.004859,   0.009169,   0.009920,   0.003129,
                    0.007081,   0.005230,   0.010609,   0.000431,   0.000460,
                    0.008990,   0.007359,   0.006778,   0.000329,   0.011602,
                    0.005150,   0.002819,   0.000629,   0.001191,   0.000690,
                    0.001089,   0.005951,   0.003071,   0.001031,   0.000639,
                    0.003030,   0.010210,   0.005150,   0.000070,   0.001861,
                    0.008061,   0.001941,   0.002510,   0.000520,   0.001079,
                    0.003011,   0.009070,   0.003640,   0.007100,   0.008371,
                    0.009629,   0.008320,   0.006241,   0.010960,   0.003059,
                    0.005689,   0.002461,   0.006529,   0.007369,   0.001631,
                    0.000760,   0.005871,   0.001307,   0.003390,   0.002621,
                    0.001060,   0.001999,   0.003330,   0.005590,   0.011739,
                    0.006660,   0.001960,   0.001960,   0.006120,   0.001810,
                    0.013680,   0.002810,   0.004361,   0.002200,   0.002321,
                    0.007621,   0.007790,   0.012030,   0.008959,   0.024079,
                    0.002120,   0.012540,   0.003480,   0.006181,   0.000469,
                    0.001360,   0.007480,   0.009370,   0.010019,   0.002860,
                    0.002589,   0.002471,   0.002720,   0.004039,   0.009000,
                    0.007330,   0.010430,   0.006280,   0.009489,   0.002299,
                    0.002781,   0.004230,   0.003071,   0.004020,   0.003461,
                    0.002180,   0.005881,   0.002611,   0.009271,   0.009240,
                    0.004600,   0.004649,   0.002860,   0.004189,   0.006411,
                    0.006500,   0.004371,   0.004840,   0.001941,   0.009840,
                    0.005619,   0.004470,   0.004419,   0.009530,   0.014890,
                    0.012920,   0.007899,   0.007899,   0.004400,   0.006660,
                    0.000990,   0.000939,   0.000990,   0.000990,   0.001041,
                    0.003560,   0.002800,   0.001510,   0.002180,   0.004419,
                    0.004221,   0.008419,   0.004189,   0.009399,   0.001650,
                    0.001989,   0.000530,   0.000501,   0.001379,   0.004779,
                    0.002200,   0.002439,   0.004690,   0.002519,   0.006561,
                    0.002321,   0.003020,   0.003269,   0.004160,   0.002130,
                    0.001711,   0.003429,   0.003260,   0.000099,   0.000731,
                    0.000070,   0.000740,   0.001421,   0.002749,   0.000431,
                    0.001270,   0.000431,   0.002016,   0.000431,   0.006048,
                    0.000520,   0.000680,   0.000680,   0.002340,   0.000929,
                    0.000929,   0.000041,   0.000029,   0.001861,   0.006941,
                    0.001169,   0.002059,   0.008349,   0.009731,   0.002270,
                    0.001237,   0.001752,   0.000210,   0.000339,   0.001159,
                    0.003161,   0.002161,   0.012040,   0.008639,   0.001909,
                    0.002270,   0.001280,   0.005651,   0.000721,   0.000721,
                    0.007550,   0.000830,   0.000830,   0.001159,   0.002439,
                    0.001670,   0.001670,   0.004600,   0.002899,   0.003291,
                    0.008301,   0.003100,   0.010500,   0.000963,   0.012199,
                    0.010590,   0.005450,   0.005960,   0.002541,   0.004061,
                    0.002299,   0.000920,   0.005111,   0.003001,   0.022080,
                    0.006781,   0.005510,   0.004070,   0.010481,   0.011120,
                    0.001721,   0.002120,   0.001721,   0.003649,   0.001500,
                    0.001810,   0.005900,   0.010909,   0.013201,   0.001980,
                    0.006120,   0.009150,   0.013530,   0.003901,   0.011761,
                    0.013770,   0.004840,   0.003700,   0.004770,   0.005951,
                    0.006290,   0.006989,   0.004799,   0.008470,   0.010641,
                    0.008741,   0.001849,   0.005960,   0.008729,   0.011461,
                    0.009511,   0.000699,   0.006270,   0.006960,   0.005929,
                    0.017910,   0.013145,   0.011541,   0.006880,   0.013291,
                    0.004189,   0.013361,   0.008821,   0.002730,   0.009680,
                    0.007739,   0.005660,   0.006459,   0.015970,   0.013141,
                    0.005629,   0.007081,   0.005581,   0.000070,   0.004201,
                    0.009564,   0.006050,   0.008230,   0.008400,   0.010191,
                    0.002030,   0.002691,   0.005760,   0.008010,   0.000750,
                    0.000610,   0.001089,   0.006870,   0.005810,   0.008230,
                    0.004479,   0.003151,   0.009661,   0.001060,   0.002439,
                    0.000511,   0.000511,   0.001019,   0.002860,   0.006200,
                    0.001089,   0.000770,   0.001309,   0.001360,   0.010769,
                    0.011180,   0.003269,   0.009440,   0.012490,   0.005080,
                    0.002321,   0.010159,   0.000651,   0.002200,   0.002420,
                    0.010721,   0.008080,   0.000770,   0.003509,   0.005261,
                    0.018450,   0.004620,   0.002950,   0.010830,   0.022559,
                    0.001890,   0.001890,   0.000721,   0.000310,   0.001331,
                    0.006880,   0.012720,   0.006398,   0.001660,   0.000542,
                    0.014999,   0.007860,   0.006149,   0.001839,   0.011810,
                    0.001019,   0.011200,   0.006389,   0.010200,   0.005150,
                    0.008071,   0.003279,   0.014929,   0.001909,   0.003049,
                    0.002398,   0.005571,   0.002200,   0.000590,   0.009169,
                    0.007069,   0.004259,   0.005881,   0.002510,   0.001210,
                    0.007451,   0.000849,   0.000479,   0.009511,   0.003700,
                    0.001401,   0.001111,   0.001740,   0.010450,   0.007040,
                    0.010740,   0.002899,   0.006200,   0.005779,   0.004080,
                    0.010159,   0.001941,   0.000731,   0.000731,   0.000731,
                    0.000731,   0.000731,   0.008429,   0.005581,   0.006851,
                    0.004070,   0.005450,   0.002560,   0.002340,   0.007710,
                    0.009740,   0.018319,   0.007739,   0.010529,   0.002420,
                    0.006730,   0.007599,   0.000849,   0.001379,   0.002960,
                    0.004777,   0.007391,   0.009431,   0.006749,   0.006110,
                    0.006241,   0.005689,   0.006241,   0.006771,   0.004830,
                    0.001689,   0.001600,   0.010019,   0.002931,   0.003630,
                    0.000390,   0.000390,   0.000390,   0.003579,   0.000390,
                    0.000390,   0.000390,   0.012170,   0.005779,   0.014450,
                    0.024200,   0.004671,   0.003991,   0.009240,   0.004661,
                    0.001631,   0.000070,   0.006921,   0.007650,   0.000361,
                    0.001060,   0.001520,   0.000939,   0.001280,   0.002749,
                    0.002749,   0.004450,   0.002059,   0.011761,   0.002369,
                    0.002161,   0.002161,   0.005191,   0.003151,   0.001740,
                    0.001111,   0.002570,   0.002340,   0.001019,   0.002130,
                    0.000409,   0.000409,   0.006820,   0.001769,   0.001571,
                    0.000440,   0.002149,   0.002899,   0.002420,   0.001089,
                    0.000699,   0.003751,   0.000939,   0.000731,   0.002360,
                    0.002360,   0.000530,   0.011761,   0.006570,   0.008840,
                    0.010926,   0.009370,   0.008107,   0.007289,   0.000390,
                    0.001520,   0.001520,   0.002120,   0.004339,   0.012901,
                    0.003870,   0.007100,   0.000939,   0.016429,   0.001600,
                    0.004939,   0.012509,   0.007480,   0.007110,   0.006389,
                    0.012490,   0.006490,   0.012199,   0.010280,   0.001631,
                    0.008821,   0.001331,   0.001791,   0.007841,   0.003870,
                    0.011059,   0.009341,   0.004039,   0.001331,   0.001331,
                    0.006880,   0.009440,   0.004281,   0.004281,   0.004479,
                    0.006389,   0.004479,   0.006411,   0.005469,   0.003071,
                    0.008901,   0.000440,   0.003020,   0.016819,   0.006050,
                    0.005300,   0.002461,   0.012509,   0.013920,   0.008949,
                    0.005929,   0.001401,   0.006200,   0.000271,   0.001191,
                    0.009680,   0.009240,   0.001089,   0.010070,   0.001740,
                    0.005929,   0.002899,   0.005929,   0.003151,   0.007592,
                    0.000530,   0.002880,   0.001379,   0.006989,   0.009511,
                    0.013361,   0.006781,   0.008477,   0.008477,   0.003543,
                    0.003151,   0.000874,   0.001048,   0.000085,   0.001922,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
                    0.000000,   0.000000,   0.000000,   0.000000])

    
    tr_ratio = np.array([1.0435,   1.0670,   1.0671,   1.1102,   1.0917,   1.0466,
                        1.0553,   1.0552,   1.0870,   1.0636,   1.0953,   1.0953,
                        1.0767,   1.0504,   1.0695,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   1.0410,   1.0105,   1.0000,   1.0000,   1.0066,
                        1.0154,   1.0252,   1.0260,   1.0512,   1.0248,   1.1116,
                        1.0368,   1.0365,   1.0306,   1.1545,   1.0938,   1.0110,
                        0.9936,   1.0310,   1.0239,   1.0173,   1.0331,   1.0494,
                        1.0394,   1.0093,   0.9825,   1.0977,   1.0380,   1.0798,
                        1.0699,   1.0454,   1.0268,   1.0067,   1.0428,   1.0417,
                        1.0007,   1.0022,   1.0320,   0.9851,   0.9831,   1.0150,
                        1.0503,   1.0535,   1.0177,   1.0640,   0.9828,   1.0058,
                        0.9893,   0.9886,   0.9899,   0.9902,   1.0350,   1.0367,
                        1.0093,   1.0284,   1.0000,   1.0049,   1.0003,   1.0920,
                        1.0668,   1.0097,   1.0049,   0.9971,   0.9962,   0.9488,
                        1.0147,   1.0052,   1.0057,   0.9919,   0.9738,   0.9781,
                        1.0116,   1.0087,   1.0119,   1.0050,   0.9991,   1.0093,
                        0.9948,   0.9905,   1.0275,   1.0059,   1.0625,   1.0581,
                        1.0069,   0.9884,   1.0620,   1.0681,   1.0647,   0.9948,
                        0.9984,   1.0155,   0.9898,   1.0015,   0.9999,   1.0346,
                        1.0336,   1.0337,   1.1215,   1.1015,   1.0010,   1.0035,
                        1.0011,   1.0036,   1.0078,   1.0256,   0.9909,   1.0513,
                        0.9223,   0.9407,   1.0115,   1.0274,   1.0143,   1.0118,
                        1.0296,   1.0162,   1.0147,   1.0132,   1.1039,   1.0942,
                        1.0960,   1.0002,   0.9811,   0.9852,   0.9302,   0.9302,
                        0.9966,   1.0659,   1.0827,   1.0205,   1.0128,   1.0280,
                        1.0447,   1.0406,   1.0406,   1.0207,   1.0288,   1.0528,
                        1.0812,   1.0810,   1.0808,   1.0100,   1.0224,   1.0460,
                        1.0519,   1.1467,   1.0452,   1.0590,   1.0659,   1.0205,
                        1.0214,   1.0850,   1.0326,   1.0246,   1.0306,   1.0300,
                        1.0846,   1.0846,   1.0389,   1.0367,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                        0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000])
    
    from_bus_nr = np.array([77,    26,    27,    39,    42,    54,    89,    90,   124,
                        148,   154,   155,   168,   182,   189,  2726,  2726,  2727,
                        2725,  2725,   126,  2728,  2728,  2729,  2729,    81,    42,
                        26,    27,   145,   134,   133,   163,    40,    39,   177,
                        202,   193,   192,   206,    42,   197,    99,   161,    44,
                        184,   120,    29,    28,    25,   119,   121,    81,    91,
                        12,    30,    24,    17,    21,    26,    17,    26,    27,
                        13,    13,    32,    23,    33,    19,    26,    27,    27,
                        25,    28,    14,     8,     8,    31,    62,    60,    66,
                        50,    58,    59,    59,    67,    51,    36,    37,    54,
                        66,    66,    61,    62,    61,    60,    46,    49,    57,
                        41,    63,    47,    52,    95,    74,   105,    78,   130,
                        106,    90,    99,    80,   115,   125,   107,   117,   118,
                        78,    77,   113,   102,   109,    90,    85,   128,   131,
                        132,   101,   124,   125,   103,   104,   111,    88,   116,
                        108,   115,   106,    97,   116,    82,   123,   123,    99,
                        100,    92,    81,   121,   122,   114,   172,   173,   175,
                        176,   175,   164,   136,   169,   151,   153,   135,   147,
                        152,   152,   174,   174,   168,   169,   168,   145,   146,
                        163,   164,   153,   158,   148,   167,   167,   175,   177,
                        180,   156,   162,   160,   170,   187,   196,   188,   190,
                        192,   201,   189,   207,   199,   185,   194,   204,   185,
                        186,   393,   394,   246,   246,   248,   249,   276,   277,
                        342,   342,   343,   354,   355,   356,   361,   369,   386,
                        385,   391,   392,   398,   399,   404,   405,   406,   407,
                        413,   465,   489,   490,   545,   574,   575,   588,   588,
                        593,   613,   617,   665,   667,   669,   696,   697,   715,
                        731,   734,   741,   759,   758,   769,   768,   778,   779,
                        807,   832,   854,   884,   901,  1001,  1002,  1091,  1092,
                        1114,  1113,  1187,  1188,  1233,  1232,  1247,  1255,  1254,
                        1286,  1285,  1288,  1363,  1351,  1353,  1388,  1394,  1432,
                        1569,  1629,  1631,  1674,  1684,  1790,  1792,  1793,  1818,
                        1838,  1837,  1858,  1859,  1929,  1950,  1951,  1974,  1976,
                        1977,  1982,  2008,  2014,  2015,  2076,  2077,  2092,  2091,
                        2093,  2093,  2103,  2145,  2153,  2154,  2170,  2171,  2177,
                        2178,  2209,  2212,  2213,  2214,  2233,  2234,  2235,  2236,
                        2238,  2237,  2238,  2251,  2252,  2257,  2319,  2320,  2423,
                        2445,  2444,  2466,  2489,  2498,  2499,  2502,  2503,  2519,
                        2521,  2520,  2527,  2528,  2593,  2594,  2595,  2660,  2680,
                        2679,  2697,  2697,  2717,  2718,  2720,  2721,  2584,  2082,
                        2117,  1928,  2064,  2377,  1263,  1273,   676,   628,   774,
                        828,   677,   741,   767,  1738,  2203,  2416,  2422,  2651,
                        2485,  2686,  2672,  2608,  2613,  2561,  2559,  2578,  1688,
                        2648,   664,  1291,   523,   664,   665,   510,  2533,   762,
                        710,   695,  2551,  2584,   728,   828,  1768,  1706,  1592,
                        760,  2670,  2456,   665,   712,  2539,   658,  2583,  2580,
                        1519,   428,   317,   565,   427,   565,   346,   345,   388,
                        476,   261,   262,   459,   480,   518,   426,   370,   425,
                        480,   462,   465,   288,   303,   330,   331,   334,   376,
                        577,   389,   338,   385,   386,   373,   390,   418,   535,
                        585,   465,   464,   454,   291,   491,   536,   272,   320,
                        568,   210,   210,   550,   559,   560,   504,   495,   545,
                        533,   527,   543,   537,   491,   492,   537,   540,   538,
                        539,   560,   560,   540,   518,   567,   528,   342,   339,
                        536,   552,   541,   496,   533,   356,   519,   351,   559,
                        546,   504,   354,   356,   525,   532,   547,   507,   543,
                        552,   549,   549,   534,   537,   519,   352,   521,   520,
                        560,   559,   537,   502,   505,   544,   551,   548,   526,
                        517,   518,   545,   342,   530,   422,   551,   508,   470,
                        358,   411,   584,   522,   411,   466,   381,   416,   253,
                        423,   451,   522,   557,   529,   484,   428,   425,   513,
                        385,   570,   556,   476,   420,   382,   420,   386,   384,
                        557,   500,   318,   390,   267,   267,   558,   559,   529,
                        454,   293,   358,   450,   513,   498,   452,   319,   570,
                        553,   584,   244,   364,   344,   498,   416,   294,   455,
                        281,   324,   313,   321,   435,   511,   332,   306,   311,
                        481,   308,   583,   308,   574,   566,   300,   574,   445,
                        445,   575,   574,   392,   314,   325,   337,   326,   329,
                        337,   312,   471,   446,   313,   229,   287,   391,   391,
                        230,   512,   332,   512,   429,   469,   469,   391,   305,
                        295,   295,   575,   511,   574,   301,   486,   431,   443,
                        415,   482,   586,   586,   573,   284,   576,   573,   449,
                        397,   573,   447,   489,   456,   375,   379,   457,   499,
                        278,   444,   487,   488,   464,   433,   443,   485,   473,
                        473,   509,   489,   441,   393,   442,   394,   233,   403,
                        232,   415,   581,   581,   578,   359,   409,   464,   409,
                        304,   394,   393,   234,   441,   283,   394,   561,   227,
                        221,   523,   335,   573,   233,   218,   479,   572,   572,
                        401,   399,   400,   414,   412,   562,   562,   562,   315,
                        263,   378,   468,   224,   400,   434,   472,   472,   461,
                        585,   483,   417,   432,   269,   399,   413,   410,   370,
                        582,   432,   371,   235,   298,   414,   413,   434,   414,
                        483,   535,   349,   478,   582,   417,   216,   226,   226,
                        270,   279,   514,   256,   571,   373,   225,   377,   477,
                        474,   280,   249,   256,   217,   571,   555,   474,   554,
                        408,   477,   387,   290,   260,   462,   271,   377,   453,
                        327,   506,   408,   555,   353,   254,   255,   271,   341,
                        493,   316,   316,   369,   369,   215,   249,   514,   215,
                        564,   564,   564,   286,   357,   604,   794,   764,   835,
                        723,   783,   737,   632,   731,   724,   736,   736,   751,
                        796,   624,   698,   793,   834,   746,   746,   789,   774,
                        746,   720,   748,   592,   704,   704,   703,   829,   829,
                        608,   608,   786,   787,   789,   789,   803,   803,   697,
                        683,   724,   692,   705,   691,   702,   697,   701,   696,
                        699,   701,   697,   682,   760,   710,   602,   695,   738,
                        705,   702,   702,   702,   685,   824,   824,   685,   686,
                        726,   823,   697,   702,   695,   646,   824,   638,   746,
                        592,   747,   726,   630,   604,   691,   764,   719,   764,
                        764,   715,   813,   813,   836,   837,   837,   832,   832,
                        719,   834,   617,   618,   755,   755,   751,   708,   826,
                        618,   617,   620,   620,   820,   642,   647,   599,   814,
                        826,   635,   634,   749,   819,   832,   648,   754,   750,
                        756,   755,   711,   711,   816,   817,   794,   637,   795,
                        689,   784,   808,   807,   807,   808,   631,   737,   626,
                        707,   812,   727,   649,   707,   707,   800,   778,   808,
                        777,   782,   782,   776,   781,   779,   721,   721,   753,
                        740,   801,   741,   716,   791,   791,   809,   809,   669,
                        792,   688,   671,   659,   722,   815,   781,   821,   833,
                        833,   779,   808,   727,   815,   818,   818,   804,   706,
                        779,   732,   732,   680,   779,   766,   801,   669,   811,
                        804,   609,   668,   670,   835,   589,   661,   800,   672,
                        658,   729,   669,   796,   795,   784,   597,   669,   839,
                        771,   771,   770,   743,   679,   768,   772,   772,   645,
                        665,   768,   752,   762,   830,   830,   802,   769,   806,
                        806,   600,   798,   798,   674,   802,   802,   735,   735,
                        730,   733,   733,   734,   622,   822,   822,   713,   810,
                        827,   827,   717,   728,   654,   656,   758,   657,   678,
                        666,   758,   825,   825,   643,   759,   718,   785,   785,
                        607,   839,   761,   780,   780,   838,   838,   797,   762,
                        805,   629,   742,   742,   831,   712,   676,   788,   802,
                        742,   830,   739,   759,   743,   656,   643,   765,   651,
                        625,  1858,  1272,   984,   983,  1017,  1016,  1021,  1020,
                        1144,  1600,   901,  1599,  1327,  1717,  1325,  1328,  1718,
                        1326,  1888,  1629,  1014,  1013,  1217,  1351,  1216,   898,
                        1859,   898,  1264,  1560,  1056,  1351,  1559,  1056,  1711,
                        1353,  1711,  1712,  1353,  1712,  1888,  1674,  1673,  1321,
                        1590,  1321,  1322,  1766,  1322,  1353,  1352,  1421,   992,
                        1353,   992,  1397,  1463,  1494,  1667,  1010,  1666,  1440,
                        1387,  1222,  1068,  1746,  1119,  1497,  1588,  1750,  1750,
                        1336,  1889,  1483,  1568,  1803,  1211,   848,  1013,  1886,
                        879,  1175,  1013,  1174,  1690,  1208,  1487,  1738,  1739,
                        1554,  1209,  1201,  1866,  1867,  1749,  1078,  1544,  1125,
                        1129,  1858,  1123,  1127,   985,  1262,  1365,  1858,  1859,
                        1242,  1510,   903,  1279,  1610,  1859,  1610,  1128,  1858,
                        1126,  1124,  1101,  1122,  1919,  1920,  1676,  1234,  1235,
                        1573,  1553,  1870,  1869,  1450,  1553,  1222,  1918,  1858,
                        1132,  1623,  1689,  1917,  1688,  1720,  1879,  1364,  1312,
                        1719,  1567,  1263,  1858,  1878,   950,  1858,   949,  1858,
                        1265,  1859,  1639,  1637,  1638,  1235,  1449,   957,   958,
                        1248,  1902,  1249,   934,  1285,  1819,  1047,  1662,  1871,
                        1812,  1695,  1926,  1408,  1426,  1048,  1715,  1747,  1820,
                        1025,  1496,  1220,  1887,  1240,  1814,   853,  1813,  1884,
                        1753,  1606,  1926,  1841,  1607,  1285,  1663,  1249,  1885,
                        1015,  1014,  1248,  1428,  1026,  1831,  1409,  1830,  1241,
                        1411,  1694,  1752,  1015,  1384,  1286,   944,   967,   942,
                        1640,   952,  1539,  1594,   951,  1537,  1482,  1756,  1868,
                        1891,  1912,   865,  1910,   856,  1587,  1433,  1433,  1195,
                        1850,  1200,  1585,  1200,  1433,  1515,  1099,  1185,  1515,
                        1099,  1406,  1451,  1404,  1083,   868,  1081,  1415,  1414,
                        1631,  1670,  1190,  1378,  1809,  1436,  1641,  1434,  1836,
                        1143,  1736,  1876,   995,  1874,   995,  1435,   940,  1084,
                        1630,  1084,  1136,  1837,  1565,  1407,   869,  1565,  1405,
                        1736,  1232,  1232,  1478,  1837,  1620,  1100,  1516,  1838,
                        1100,  1514,  1196,  1909,  1906,  1194,  1908,  1252,  1834,
                        1252,   924,  1883,   922,  1271,  1717,  1271,  1318,  1718,
                        1317,  1362,  1362,  1140,  1362,  1138,  1652,  1141,  1363,
                        1651,  1137,  1033,  1419,  1032,  1151,  1381,  1809,  1611,
                        1340,  1341,  1839,  1840,  1622,  1840,  1621,  1839,  1253,
                        1470,  1848,  1253,  1468,  1296,  1784,  1822,  1782,  1785,
                        1598,  1783,  1839,  1232,  1412,  1413,  1029,  1717,  1029,
                        1379,   976,  1839,   888,  1873,   888,  1239,  1618,  1238,
                        1233,  1800,   929,  1798,  1105,  1269,  1103,  1810,  1551,
                        894,  1551,  1776,   889,  1873,   889,  1233,  1446,  1796,
                        1446,  1445,  1358,  1287,  1445,  1358,  1776,  1899,  1150,
                        1898,  1373,  1810,  1373,  1040,  1061,  1039,  1732,  1901,
                        1751,  1838,  1839,  1469,  1295,  1467,  1794,  1511,  1648,
                        1704,  1840,  1646,  1702,  1584,  1372,  1049,  1372,  1779,
                        1836,  1778,  1707,  1491,  1363,  1489,  1142,  1490,  1139,
                        1538,  1793,   982,  1390,  1431,  1440,  1431,  1402,  1730,
                        1223,  1222,  1223,  1388,  1681,  1679,  1222,  1222,   999,
                        1222,  1052,  1219,  1457,   978,  1767,  1904,  1479,  1480,
                        1387,   861,  1915,  1664,  1665,  1697,  1202,  1713,  1700,
                        1803,  1700,  1701,  1672,  1701,  1804,  1035,  1673,  1674,
                        1855,  1856,  1506,  1674,  1505,  1673,  1672,  1222,  1773,
                        1727,  1352,  1724,  1352,  1724,  1530,  1352,  1530,  1366,
                        1000,  1071,  1521,  1261,  1466,  1439,  1260,  1465,  1237,
                        1422,  1184,   973,  1183,  1188,  1832,  1218,  1396,  1223,
                        1396,   875,   876,  1176,  1300,  1172,  1173,  1193,  1854,
                        1192,  1486,  1485,  1504,  1486,  1728,  1495,  1723,  1110,
                        1721,  1315,  1023,  1398,  1022,  1804,   912,   913,   914,
                        915,  1352,  1219,  1439,  1916,  1420,  1696,  1423,  1842,
                        1803,  1853,  1389,  1833,  1442,  1804,  1442,  1092,  1658,
                        1895,  1657,  1345,  1745,  1735,  1895,  1343,  1091,  1346,
                        966,  1344,   881,  1111,   881,  1111,  1247,  1246,  1092,
                        1117,  1500,  1053,  1054,  1534,  1894,  1534,  1895,  1533,
                        1298,  1533,  1292,  1888,  1889,  1889,   927,  1888,  1889,
                        1825,  1342,  1091,  1333,  1826,  1501,  1556,   891,  1108,
                        1577,  1734,  1807,  1001,  1002,  1075,  1002,   963,  1759,
                        1758,  1555,  1578,  1500,  1057,  1520,  1092,  1518,  1616,
                        1149,  1615,  1677,  1517,  1755,  1881,  1001,  1500,  1501,
                        1331,   971,  1788,  1112,  1149,   961,   906,  1399,  1400,
                        1401,  1685,  1168,  1290,  1166,  1684,  1684,  1685,  1684,
                        1705,  1169,  1817,  1169,  1818,  1817,  1818,  1818,  1817,
                        1824,  1376,  1818,  1393,  1393,  1394,  1394,  1393,  1818,
                        1393,  1393,   994,  1558,  1823,  1541,   956,  1684,  1691,
                        1691,  1690,  1228,  1686,  1226,  1229,  1687,  1227,  1691,
                        1685,  1438,  1182,  1774,  1644,  1078,  1077,  1604,  1374,
                        1096,  1738,  1738,  1739,   893,  1775,   892,  1077,  1077,
                        1843,  1571,   997,  1289,  1817,  1818,  1893,   917,  1078,
                        1393,  1074,  1613,  1455,  1527,  1330,  1448,  1528,  1775,
                        1775,  1394,  1635,  1774,  1375,  1385,  1386,   883,  1576,
                        1844,  1536,  1303,  1852,   998,  1308,  1447,  1078,  1009,
                        1274,  1790,  1764,  1765,  1764,  1765,  1765,  1791,   930,
                        1255,   874,  1011,   872,  1764,  1786,  1120,  1790,  1790,
                        1742,  1743,  1121,  1863,  1474,  1791,  1245,  1591,  1602,
                        1254,  1254,  1498,  1582,  1255,  1094,  1653,  1094,  1806,
                        1765,  1350,  1787,   959,  1791,  1790,  1838,  1276,  1359,
                        1626,  1357,  1597,   910,  1627,   911,  1631,  1566,  1737,
                        1566,   928,   929,  1838,  1643,  1620,  1642,  1412,  1066,
                        1552,  1617,  1550,  1760,  1922,  1114,  1114,  1708,  1310,
                        1415,  1309,   867,   867,   866,   925,   937,  1106,  1270,
                        923,   937,  1104,  1425,  1872,  1425,  1043,  1717,  1199,
                        1849,  1197,  1633,  1802,   948,  1526,  1546,  1847,  1836,
                        1589,   920,  1837,  1682,  1631,  1816,  1145,  1815,  1259,
                        895,  1877,  1596,   929,  1875,  1596,  1368,  1829,   975,
                        1828,  1857,  1232,   908,   909,  1203,  1363,  1432,  1523,
                        1821,  1115,  1647,  1044,  1324,  1191,  1731,  1189,  1512,
                        1780,  1363,   856,   857,  1872,  1586,  1503,   858,   859,
                        1089,  1090,  1280,  1281,  1733,  1037,  1038,  1030,  1031,
                        1232,  1797,  1339,  1925,  1230,  1923,   928,  1991,  2307,
                        2434,  2434,  2272,  2180,  1995,  2380,  1990,  2136,  1989,
                        2274,  2316,  2395,  2282,  2317,  2310,  2126,  1961,  2302,
                        2425,  1995,  2123,  2381,  2396,  2362,  2362,  2068,  2327,
                        2329,  2046,  2217,  2216,  2394,  2148,  2203,  2203,  2132,
                        2391,  2323,  2269,  2270,  2376,  2387,  2378,  2378,  2386,
                        2379,  2414,  2323,  2385,  2265,  2265,  2392,  2384,  2384,
                        2392,  2369,  2369,  2212,  2333,  2213,  2415,  2366,  2280,
                        2050,  2333,  2437,  2390,  2405,  2204,  2405,  2077,  2044,
                        2375,  2375,  2389,  2390,  2390,  2374,  2371,  2370,  2371,
                        2309,  2354,  2389,  2077,  1939,  2446,  1939,  2346,  2027,
                        2223,  2246,  2372,  2438,  2381,  2332,  1984,  2233,  2404,
                        2404,  2404,  2183,  2001,  2349,  2349,  2234,  1949,  1982,
                        2352,  2002,  2427,  1932,  2408,  2408,  2408,  2436,  2173,
                        2202,  2303,  2303,  2067,  2179,  2259,  2231,  2199,  2232,
                        2205,  2427,  2151,  2273,  2308,  2193,  2255,  1983,  2249,
                        2003,  2318,  2372,  2410,  2410,  2410,  2372,  2034,  2306,
                        2237,  2381,  2292,  2292,  1960,  2318,  2307,  2234,  2012,
                        2011,  2308,  2151,  2436,  2291,  2332,  2250,  2365,  1960,
                        2250,  2226,  2232,  2227,  2232,  2279,  2266,  2318,  2411,
                        1949,  2427,  2238,  2237,  2238,  2155,  2258,  2238,  2256,
                        2186,  2258,  2352,  2438,  2349,  2250,  2352,  2125,  2258,
                        2257,  2435,  1966,  2063,  2327,  2211,  2398,  2398,  2398,
                        2019,  2211,  2210,  2195,  2334,  2156,  2156,  2085,  2091,
                        2397,  2069,  2088,  2053,  2066,  2215,  2094,  2268,  2210,
                        2209,  2305,  2351,  2350,  2142,  2090,  2092,  1930,  1931,
                        2100,  2089,  2133,  2063,  1935,  2100,  2055,  1930,  2210,
                        2268,  2126,  2057,  2117,  2208,  2310,  2356,  2377,  2137,
                        2065,  2065,  2220,  2356,  2124,  2276,  2326,  2326,  2298,
                        2187,  2433,  2126,  2298,  2298,  2136,  2118,  2296,  2315,
                        2118,  2315,  2406,  1943,  2181,  2403,  2317,  2158,  2049,
                        2283,  2296,  2402,  2157,  1989,  2283,  2026,  2025,  2018,
                        2030,  2017,  2177,  2413,  1985,  2426,  2447,  2447,  2325,
                        1959,  2022,  2253,  2254,  2330,  2236,  2294,  2428,  2178,
                        2196,  2196,  2278,  2325,  2383,  2277,  2166,  2235,  2022,
                        2419,  2190,  2353,  2345,  1940,  2278,  2139,  1940,  2191,
                        2345,  2328,  2244,  2278,  2086,  2419,  2383,  2324,  2331,
                        2245,  2272,  2324,  2353,  2293,  2165,  2177,  1968,  2178,
                        2357,  2278,  2428,  2423,  2271,  2393,  2393,  1947,  2424,
                        2300,  2423,  2320,  2347,  2319,  2319,  2319,  2320,  2267,
                        1947,  2289,  2313,  1950,  2319,  2221,  2161,  2320,  1951,
                        2358,  2062,  2358,  2281,  2297,  2163,  2297,  2163,  2297,
                        2071,  1996,  2127,  2143,  2046,  1999,  2046,  2134,  2432,
                        1978,  2170,  2171,  2343,  2343,  2170,  2171,  2339,  2344,
                        2343,  2343,  2340,  2106,  2364,  2363,  1946,  2248,  2432,
                        2160,  2275,  1958,  2360,  2059,  2360,  2361,  2059,  2095,
                        2218,  2219,  2169,  2169,  2095,  2161,  2141,  2341,  2342,
                        2295,  2083,  2079,  2407,  2080,  2400,  2130,  2409,  2311,
                        2263,  2096,  2078,  2097,  2109,  2098,  2439,  2242,  2252,
                        2251,  2242,  2252,  2114,  2239,  2240,  2252,  2401,  2149,
                        2147,  2366,  2288,  2104,  2112,  2112,  2399,  2099,  2102,
                        2122,  2121,  2105,  2443,  2275,  2217,  2150,  2442,  2039,
                        2040,  2216,  2036,  2035,  2444,  2439,  2444,  2445,  2445,
                        2445,  2121,  2444,  1953,  2444,  2109,  2216,  1972,  1976,
                        1977,  2262,  2412,  1977,  2312,  2081,  2288,  2280,  2264,
                        2084,  2021,  2075,  1997,  2087,  2430,  2431,  2241,  2429,
                        2431,  2159,  2028,  2284,  2285,  2153,  2285,  2382,  2420,
                        2028,  2154,  2192,  2418,  2417,  2417,  2194,  2154,  1986,
                        2418,  2382,  2189,  2194,  2420,  1954,  2664,  2570,  2675,
                        2629,  2563,  2654,  2453,  2649,  2473,  2576,  2554,  2524,
                        2554,  2613,  2722,  2523,  2585,  2478,  2491,  2491,  2695,
                        2586,  2706,  2690,  2612,  2477,  2602,  2649,  2524,  2469,
                        2688,  2571,  2609,  2588,  2587,  2451,  2545,  2526,  2536,
                        2526,  2583,  2451,  2646,  2646,  2464,  2708,  2708,  2708,
                        2458,  2582,  2509,  2553,  2455,  2642,  2543,  2652,  2474,
                        2630,  2719,  2675,  2686,  2609,  2529,  2463,  2450,  2631,
                        2716,  2534,  2469,  2608,  2624,  2646,  2450,  2604,  2641,
                        2626,  2495,  2496,  2709,  2638,  2548,  2677,  2462,  2677,
                        2487,  2700,  2532,  2566,  2698,  2682,  2585,  2683,  2510,
                        2692,  2627,  2644,  2569,  2692,  2461,  2456,  2676,  2508,
                        2623,  2486,  2639,  2473,  2685,  2697,  2684,  2645,  2664,
                        2456,  2664,  2518,  2475,  2486,  2682,  2623,  2703,  2702,
                        2558,  2525,  2560,  2555,  2472,  2561,  2611,  2645,  2697,
                        2697,  2514,  2515,  2687,  2689,  2689,  2555,  2687,  2699,
                        2664,  2668,  2661,  2658,  2662,  2663,  2589,  2659,  2563,
                        2482,  2662,  2564,  2660,  2653,  2573,  2603,  2467,  2497,
                        2483,  2695,  2497,  2481,  2478,  2501,  2506,  2470,  2538,
                        2696,  2567,  2567,  2567,  2505,  2513,  2671,  2506,  2470,
                        2694,  2678,  2667,  2500,  2610,  2635,  2707,  2665,  2704,
                        2636,  2693,  2505,  2618,  2500,  2512,  2506,  2479,  2502,
                        2503,  2504,  2575,  2678,  2616,  2678,  2565,  2673,  2621,
                        2448,  2666,  2590,  2622,  2640,  2632,  2633,  2499,  2632,
                        2499,  2597,  2498,  2517,  2628,  2471,  2681,  2694,  2540,
                        2591,  2541,  2669,  2599,  2477,  2537,  2535,  2657,  2656,
                        2667,  2457,  2643,  2500,  2501,  2617,  2468,  2580,  2705,
                        2452,  2460,  2670,  2607,  2507,  2568,  2484,  2449,  2516,
                        2549,  2551,  2596,  2530,  2562,  2601,  2605,  2598,  2580,
                        2606,  2581,  2584,  2601,  2600,  2607,  2581,  2637,  2592,
                        2593,  2530,  2453,  2531,  2511,  2542,  2547,  2655,  2544,
                        2546,  2488,  2714,  2557,  2557,  2620,  2714,  2691,  2557,
                        2539,  2542,  2480,  2648,  2454,  2620,  2494,  2634,  2572,
                        2552,  2459,  2625,  2619,  2602,  2715,  2493,  2556,  2574,
                        2674,  2554,  2614,  2730,  2730,  2735,  2736,  2732,  2734,
                        2731,  2733,     5,     7,     9,    11,    12,    16,    18,
                        20,    22,    29,    37,    40,    43,    45,    48,    51,
                        53,    55,    57,    59,    62,    64,    72,    74,    76,
                        78,    80,    84,    86,    88,    90,    92,    94,    96,
                        98,   100,   102,   104,   106,   108,   111,   113,   116,
                        118,   120,   122,   125,   127,   129,   134,   138,   141,
                        144,   146,   148,   150,   153,   155,   157,   159,   162,
                        164,   165,   169,   171,   173,   176,   178,   184,   186,
                        188,   190,   193,   195,   197,   199,   201,   203,   205,
                        207,   213,   221,   228,   243,   249,   258,   265,   282,
                        284,   286,   298,   311,   319,   326,   335,   343,   345,
                        355,   356,   362,   376,   380,   386,   394,   405,   407,
                        422,   430,   437,   439,   442,   458,   465,   476,   486,
                        488,   490,   495,   508,   510,   516,   518,   521,   525,
                        531,   533,   537,   539,   542,   548,   552,   557,   575,
                        578,   588,   594,   614,   618,   665,   686,   697,   715,
                        731,   741,   759,   769,   779,   808,   847,   850,   852,
                        853,   855,   857,   861,   864,   866,   869,   871,   873,
                        876,   878,   883,   885,   891,   895,   897,   900,   901,
                        899,   903,   905,   907,   911,   917,   919,   921,   923,
                        927,   929,   931,   933,   935,   939,   941,   943,   946,
                        948,   954,   956,   960,   962,   964,   966,   968,   970,
                        972,   974,   976,   978,   980,   982,   986,   988,   990,
                        994,   996,   998,  1000,  1001,  1004,  1006,  1008,  1010,
                        1012,  1014,  1015,  1013,  1019,  1025,  1026,  1036,  1042,
                        1044,  1046,  1048,  1050,  1052,  1058,  1060,  1062,  1065,
                        1068,  1070,  1072,  1074,  1076,  1078,  1080,  1082,  1086,
                        1088,  1092,  1096,  1102,  1104,  1108,  1110,  1112,  1114,
                        1115,  1113,  1117,  1119,  1123,  1133,  1135,  1138,  1139,
                        1144,  1145,  1149,  1151,  1156,  1165,  1167,  1171,  1173,
                        1178,  1180,  1182,  1186,  1188,  1195,  1202,  1205,  1207,
                        1209,  1211,  1213,  1215,  1219,  1221,  1223,  1225,  1231,
                        1233,  1235,  1237,  1241,  1243,  1245,  1247,  1249,  1255,
                        1257,  1259,  1263,  1265,  1267,  1269,  1270,  1277,  1279,
                        1283,  1286,  1288,  1290,  1292,  1294,  1296,  1298,  1300,
                        1302,  1304,  1306,  1308,  1312,  1314,  1316,  1320,  1324,
                        1330,  1332,  1334,  1336,  1338,  1343,  1345,  1348,  1350,
                        1352,  1353,  1351,  1355,  1357,  1361,  1363,  1365,  1367,
                        1369,  1375,  1376,  1374,  1378,  1380,  1382,  1384,  1386,
                        1388,  1390,  1392,  1394,  1398,  1403,  1409,  1411,  1413,
                        1415,  1417,  1419,  1421,  1423,  1427,  1429,  1433,  1438,
                        1440,  1444,  1448,  1450,  1452,  1454,  1456,  1458,  1460,
                        1462,  1464,  1468,  1472,  1474,  1476,  1478,  1480,  1482,
                        1484,  1486,  1488,  1490,  1493,  1495,  1497,  1499,  1501,
                        1503,  1505,  1510,  1512,  1518,  1520,  1522,  1524,  1526,
                        1528,  1536,  1538,  1542,  1544,  1546,  1548,  1554,  1556,
                        1558,  1562,  1564,  1568,  1570,  1572,  1574,  1576,  1578,
                        1580,  1582,  1584,  1586,  1588,  1590,  1592,  1594,  1598,
                        1602,  1604,  1606,  1608,  1612,  1614,  1618,  1620,  1624,
                        1626,  1628,  1630,  1631,  1629,  1633,  1635,  1637,  1641,
                        1645,  1647,  1650,  1654,  1660,  1663,  1665,  1669,  1671,
                        1673,  1674,  1672,  1676,  1678,  1680,  1683,  1685,  1687,
                        1689,  1691,  1695,  1697,  1699,  1703,  1706,  1708,  1710,
                        1714,  1716,  1718,  1720,  1726,  1728,  1730,  1732,  1735,
                        1737,  1739,  1745,  1747,  1749,  1751,  1753,  1755,  1757,
                        1761,  1763,  1765,  1767,  1769,  1771,  1773,  1775,  1777,
                        1781,  1787,  1789,  1791,  1792,  1795,  1797,  1802,  1804,
                        1806,  1808,  1810,  1812,  1818,  1820,  1822,  1824,  1826,
                        1828,  1831,  1833,  1835,  1837,  1838,  1836,  1840,  1844,
                        1846,  1848,  1850,  1852,  1854,  1859,  1861,  1863,  1865,
                        1867,  1873,  1879,  1881,  1883,  1885,  1887,  1889,  1891,
                        1893,  1895,  1897,  1901,  1903,  1905,  1907,  1911,  1914,
                        1916,  1918,  1922,  1924,  1927,  1929,  1937,  1942,  1945,
                        1948,  1951,  1958,  1963,  1965,  1975,  1977,  1983,  1992,
                        1994,  1998,  2000,  2002,  2006,  2009,  2012,  2015,  2018,
                        2024,  2033,  2038,  2042,  2044,  2047,  2060,  2064,  2069,
                        2071,  2074,  2077,  2091,  2102,  2106,  2108,  2111,  2113,
                        2120,  2122,  2128,  2130,  2132,  2135,  2138,  2144,  2146,
                        2148,  2154,  2158,  2171,  2176,  2178,  2184,  2198,  2207,
                        2210,  2213,  2215,  2217,  2219,  2222,  2225,  2230,  2234,
                        2236,  2238,  2247,  2252,  2254,  2258,  2261,  2263,  2274,
                        2278,  2288,  2290,  2294,  2312,  2314,  2320,  2342,  2344,
                        2348,  2351,  2355,  2359,  2361,  2364,  2367,  2369,  2371,
                        2374,  2376,  2379,  2385,  2387,  2390,  2422,  2424,  2441,
                        2443,  2445,  2454,  2466,  2471,  2490,  2492,  2499,  2501,
                        2503,  2506,  2520,  2521,  2528,  2534,  2588,  2593,  2595,
                        2609,  2615,  2630,  2651,  2661,  2663,  2680,  2698,  2702,
                        2718,  2721,  2734])
    
    to_bus_nr = np.array([8,    28,    29,    41,    45,    56,    91,    92,   127,
                        149,   156,   157,   170,   183,   191,   156,   157,    67,
                        75,   104,  2724,   147,   148,    47,    48,  2723,    23,
                        89,    90,    30,    33,    20,    21,   108,    90,    82,
                        164,   164,   163,   166,    10,    18,    54,    81,     9,
                        149,    56,    92,   122,   191,    63,    63,    31,    28,
                        4,    11,     4,     7,    10,     5,     3,    19,    20,     6,
                        12,    13,    11,     5,     4,     1,     2,    24,     9,
                        25,    29,    15,    14,    29,    58,    55,    34,    34,
                        42,    43,    34,    51,    38,    55,    35,    37,    36,
                        65,    42,    43,    39,    40,    39,    45,    53,    57,
                        57,    64,    44,   124,   105,    93,    84,   129,   130,
                        130,   118,   112,   100,    73,    89,    78,    77,    75,
                        72,    77,    71,   116,    68,    89,    86,    75,    76,
                        76,   103,   110,    83,    94,    79,    77,   107,    98,
                        78,    87,    78,    77,    96,   117,   109,    70,    69,
                        126,   126,   114,   120,   127,   155,   154,   154,   155,
                        143,   139,   154,   172,   173,   155,   136,   142,   180,
                        179,   146,   168,   139,   166,   151,   134,   133,   146,
                        145,   142,   147,   140,   140,   158,   135,   176,   172,
                        137,   138,   161,   150,   192,   202,   207,   193,   181,
                        203,   200,   182,   183,   191,   186,   198,   204,   205,
                        1,     2,   245,    29,     3,     3,     4,     5,     6,     6,
                        9,    10,    11,    12,    14,    15,    17,    18,    19,    20,
                        21,    21,    23,    23,    24,    24,    25,    30,    31,
                        31,    32,    33,    33,    34,    34,    35,    37,    38,
                        42,    46,    48,    49,    49,    50,    53,    52,    55,
                        60,    60,    61,    62,    63,    64,    65,    67,    68,
                        71,    74,    81,    81,    82,    82,    83,    84,    87,
                        88,    93,    94,    95,    99,   100,   101,   102,   103,
                        104,   105,   106,   107,   109,   110,   112,   114,   114,
                        115,   118,   119,   121,   122,   123,   124,   125,   128,
                        129,   133,   135,   135,   136,   137,   138,   139,   140,
                        142,   142,   143,   143,   145,   146,  2091,   146,   147,
                        151,   152,   153,   155,   154,   158,   158,   160,   161,
                        161,   163,   166,   166,   167,   167,   169,   170,   170,
                        172,   173,   174,   175,   176,   177,   180,   180,   181,
                        184,   185,   186,   188,   187,   189,   190,   191,   192,
                        193,   195,   196,   197,   198,   201,   200,   202,   203,
                        204,   204,   206,   207,   349,   292,   315,   419,   220,
                        523,   242,   439,   380,   405,   463,  1279,  1575,  1654,
                        1361,   597,  1149,  1148,   984,  2152,  2357,  2011,  2214,
                        2215,  2214,  2091,  1933,  2215,   790,  2202,   452,  2074,
                        1213,   463,   252,  1017,  2388,   380,   333,   452,   254,
                        240,   479,  1668,   709,   709,   740,   333,   372,   336,
                        470,   448,  2331,  1021,  2388,   347,  2134,   356,   219,
                        518,   565,   569,   342,   343,   342,   343,   404,   405,
                        343,   404,   459,   348,   239,   237,   273,   450,   253,
                        276,   277,   392,   330,   259,   277,   430,   315,   238,
                        327,   373,   289,   354,   374,   465,   328,   222,   478,
                        426,   402,   455,   294,   320,   352,   517,   354,   361,
                        354,   503,   504,   494,   355,   357,   355,   354,   532,
                        563,   354,   354,   526,   550,   537,   536,   544,   501,
                        503,   343,   275,   275,   274,   274,   339,   340,   496,
                        563,   563,   362,   362,   507,   351,   272,   421,   579,
                        516,   545,   527,   492,   580,   536,   501,   532,   536,
                        533,   534,   580,   508,   579,   515,   546,   219,   505,
                        340,   502,   518,   517,   404,   405,   497,   402,   542,
                        567,   524,   531,   528,   361,   333,   296,   344,   466,
                        427,   252,   450,   239,   381,   231,   231,   450,   436,
                        437,   522,   211,   424,   386,   239,   383,   383,   553,
                        293,   348,   385,   382,   384,   338,   338,   338,   374,
                        222,   424,   423,   211,   497,   484,   239,   282,   345,
                        296,   318,   239,   451,   317,   396,   396,   500,   385,
                        244,   475,   291,   374,   404,   405,   388,   314,   276,
                        307,   276,   277,   276,   241,   241,   311,   312,   310,
                        583,   310,   329,   566,   566,   309,   392,   481,   446,
                        236,   236,   309,   277,   309,   326,   306,   300,   259,
                        471,   307,   574,   276,   287,   350,   575,   313,   322,
                        322,   350,   392,   324,   321,   391,   391,   305,   389,
                        314,   435,   575,   406,   258,   431,   407,   465,   218,
                        359,   331,   234,   467,   449,   419,   419,   299,   299,
                        456,   447,   288,   486,   375,   458,   467,   485,   444,
                        486,   328,   376,   438,   223,   407,   223,   490,   243,
                        394,   232,   264,   212,   379,   363,   213,   212,   433,
                        448,   213,   273,   406,   228,   393,   487,   264,   265,
                        245,   245,   265,   561,   257,   499,   278,   490,   328,
                        576,   363,   482,   403,   523,   278,   412,   395,   365,
                        366,   268,   297,   268,   460,   298,   237,   263,   413,
                        468,   251,   371,   395,   418,   298,   297,   240,   398,
                        398,   400,   368,   367,   360,   237,   240,   412,   235,
                        237,   292,   360,   400,   413,   398,   336,   269,   378,
                        251,   224,   410,   214,   369,   208,   226,   217,   279,
                        255,   327,   323,   289,   247,   248,   209,   327,   247,
                        249,   214,   555,   302,   440,   248,   372,   209,   353,
                        217,   250,   208,   250,   271,   271,   323,   260,   289,
                        238,   290,   554,   387,   440,   214,   214,   270,   341,
                        302,   217,   369,   225,   280,   216,   245,   285,   266,
                        561,   355,   839,   626,   590,   605,   598,   730,   731,
                        629,   729,   617,   730,   731,   680,   603,   594,   823,
                        624,   641,   745,   621,   587,   621,   662,   662,   720,
                        748,   588,   700,   700,   703,   587,   587,   605,   696,
                        786,   787,   590,   590,   595,   683,   595,   595,   696,
                        692,   690,   690,   686,   684,   693,   693,   699,   682,
                        653,   653,   602,   601,   601,   696,   588,   696,   697,
                        588,   587,   601,   823,   588,   684,   605,   738,   693,
                        691,   638,   638,   646,   601,   650,   588,   748,   747,
                        696,   745,   705,   617,   642,   618,   675,   675,   714,
                        598,   714,   832,   836,   715,   817,   681,   715,   610,
                        611,   719,   603,   755,   618,   708,   612,   615,   615,
                        612,   816,   647,   794,   598,   599,   698,   832,   635,
                        634,   749,   819,   832,   755,   754,   750,   756,   755,
                        755,   681,   816,   814,   593,   689,   687,   687,   773,
                        723,   783,   631,   627,   627,   773,   706,   807,   649,
                        613,   613,   614,   775,   775,   594,   594,   594,   776,
                        766,   778,   652,   652,   614,   707,   632,   741,   716,
                        614,   594,   767,   593,   671,   591,   591,   792,   633,
                        633,   659,   613,   763,   796,   821,   792,   593,   641,
                        811,   636,   636,   631,   792,   614,   680,   753,   740,
                        637,   763,   763,   632,   609,   812,   793,   591,   671,
                        668,   808,   737,   732,   777,   670,   722,   661,   672,
                        688,   778,   626,   722,   639,   768,   769,   770,   744,
                        679,   665,   761,   761,   765,   622,   622,   616,   616,
                        752,   788,   600,   768,   640,   640,   802,   802,   600,
                        673,   673,   730,   663,   663,   731,   694,   730,   730,
                        733,   596,   664,   713,   628,   600,   810,   667,   667,
                        717,   667,   655,   655,   667,   657,   678,   717,   759,
                        644,   644,   718,   619,   619,   606,   606,   694,   757,
                        769,   757,   600,   797,   739,   623,   607,   805,   606,
                        831,   677,   596,   674,   769,   725,   660,   799,   799,
                        651,   744,   654,   790,   645,   666,   831,  1273,  1273,
                        1492,   984,   986,  1017,  1360,  1021,   900,  1348,  1600,
                        1600,  1247,  1327,  1327,  1246,  1328,  1328,  1630,  1301,
                        884,   885,  1632,  1217,  1217,  1412,   898,   896,  1413,
                        1206,  1560,  1056,  1560,  1055,   975,  1711,  1709,   976,
                        1712,  1710,  1364,  1307,  1851,  1180,  1321,  1319,   928,
                        1322,  1320,   840,   841,  1007,  1008,   992,   991,  1233,
                        1337,  1781,  1863,  1667,  1667,  1771,  1293,  1449,   990,
                        1502,  1432,  1432,  1313,   962,  1332,  1247,  1004,  1267,
                        1267,  1015,  1013,  1804,  1690,   939,  1215,  1714,  1175,
                        1175,  1224,   970,  1645,  1805,   960,  1388,   884,   885,
                        884,   885,  1791,  1790,  1429,  1102,  1125,  1129,  1125,
                        1129,   855,  1234,  1135,  1204,  1134,   854,  1262,  1234,
                        1205,  1130,  1610,  1609,  1131,  1128,  1128,   855,  1124,
                        1124,   854,   855,   854,  1740,  1741,  1416,  1416,  1417,
                        1870,  1869,  1870,  1869,  1294,  1133,   854,   855,  1278,
                        1689,  1669,  1484,  1263,  1311,   897,  1003,  1509,  1212,
                        1636,  1243,  1235,   950,   950,   902,  1859,  1574,  1624,
                        1639,  1639,  1675,  1553,   854,   855,   933,   932,   851,
                        1695,  1221,  1285,   935,  1024,  1428,  1871,  1811,  1725,
                        1314,   846,  1410,  1286,  1694,   847,  1286,  1695,  1569,
                        1014,  1014,  1570,  1814,  1814,  1225,  1884,  1285,  1285,
                        1812,  1927,  1027,  1927,  1716,  1569,  1569,  1570,  1569,
                        1118,  1427,  1726,  1605,  1608,  1903,  1286,  1285,   879,
                        1383,   969,   989,  1006,   944,   944,  1335,  1640,   952,
                        1539,   952,  1539,  1067,  1363,   921,  1868,  1868,  1912,
                        1912,  1115,  1287,  1545,  1194,   921,  1432,  1432,  1200,
                        1198,  1186,  1170,  1515,  1099,  1513,  1097,  1630,  1406,
                        1406,  1629,  1083,  1083,  1630,  1629,  1761,  1718,  1232,
                        1477,  1794,  1619,  1436,  1436,  1640,   947,  1283,  1475,
                        1876,  1876,  1612,  1619,  1369,  1418,  1084,  1082,   975,
                        1323,  1737,  1565,  1407,  1563,  1407,  1145,  1827,  1034,
                        1005,  1471,  1838,   980,  1100,  1516,  1098,  1516,   920,
                        1196,  1909,  1196,  1909,  1114,  1252,  1250,  1114,   924,
                        924,  1114,  1271,  1268,  1114,  1318,  1318,  1288,  1288,
                        1113,  1140,  1140,  1115,  1652,  1141,  1652,  1141,  1873,
                        1033,  1033,  1525,   975,  1795,   928,   928,   929,  1655,
                        1656,  1113,  1622,  1622,  1113,  1113,  1253,  1470,  1251,
                        1470,  1113,  1412,  1784,  1784,  1413,  1785,  1785,  1703,
                        1007,  1146,  1147,  1628,  1029,  1028,  1625,  1277,  1369,
                        1414,   888,   886,  1415,  1239,  1239,  1799,  1232,  1800,
                        1800,  1115,  1105,  1105,  1900,   849,  1551,  1549,  1019,
                        1777,   889,   887,  1770,  1287,  1446,  1444,  1050,  1445,
                        1358,  1443,  1356,  1414,  1840,  1899,  1899,  1583,  1373,
                        1371,  1233,  1040,  1040,  1649,  1650,  1593,   920,  1041,
                        1042,  1469,  1469,  1512,  1288,  1288,  1648,  1704,  1648,
                        1704,   885,   884,  1372,  1370,  1579,  1779,  1779,  1619,
                        943,  1491,  1491,  1362,  1142,  1142,  1751,  1864,   946,
                        945,  1179,  1431,  1430,  1179,  1351,  1351,  1865,  1914,
                        1913,  1729,  1681,  1681,   860,  1316,  1051,  1680,  1179,
                        1179,  1210,  1299,  1896,  1387,  1388,  1897,  1905,  1388,
                        1659,   945,  1180,  1177,  1180,  1178,  1700,  1698,  1177,
                        1701,  1699,  1672,  1672,   945,   946,  1674,  1672,  1486,
                        1506,  1506,  1485,  1845,   918,   877,   877,  1109,  1072,
                        1724,  1722,   878,  1530,  1529,  1352,  1440,  1548,  1439,
                        1218,  1261,  1466,  1261,  1466,  1218,  1188,  1219,  1184,
                        1184,  1547,  1367,  1458,  1793,  1396,  1395,  1180,   954,
                        953,  1176,  1176,  1176,  1485,  1193,  1193,  1660,  1284,
                        1485,  1257,  1522,  1439,   974,  1723,  1723,   878,   878,
                        1023,  1023,   977,   945,   946,   946,   945,  1772,   981,
                        1403,   919,  1353,  1178,  1236,  1793,  1036,  1769,  1792,
                        1792,  1846,  1442,  1441,  1079,  1111,  1658,  1658,  1894,
                        899,   899,  1069,   899,  1148,   900,  1346,  1346,  1058,
                        881,   880,  1493,   900,   901,  1086,  1091,  1091,  1092,
                        1091,  1092,  1534,  1532,  1297,  1091,  1533,  1531,  1267,
                        1302,  1500,  1501,  1334,  1692,  1693,  1266,  1894,  1070,
                        1266,  1454,   964,  1453,  1001,   899,  1064,  1500,  1080,
                        1092,  1754,  1085,  1076,  1744,  1085,  1086,   890,   906,
                        926,  1079,  1111,  1789,  1678,  1001,  1616,  1616,  1002,
                        1148,  1091,  1116,  1880,  1001,  1002,  1065,   901,  1107,
                        965,  1079,   972,  1112,  1500,  1501,  1500,   956,  1394,
                        1168,  1168,   916,   863,  1572,  1275,  1394,  1394,  1169,
                        1167,  1614,  1162,  1163,  1157,  1158,  1572,  1571,  1155,
                        1152,  1153,  1154,  1159,  1160,  1165,  1156,   870,   871,
                        1073,  1374,  1376,  1542,  1762,   938,  1684,  1685,  1774,
                        1228,  1228,  1775,  1229,  1229,  1603,  1437,  1391,  1214,
                        1354,   882,  1540,  1306,  1355,   993,  1739,  1063,  1860,
                        1861,  1461,   893,   893,  1392,   905,   955,   864,  1374,
                        1684,  1375,  1376,  1572,  1535,  1095,  1164,  1571,  1374,
                        1077,  1456,  1376,  1305,  1306,  1690,  1684,  1161,   998,
                        1181,  1557,  1571,  1572,   904,  1394,  1464,   955,  1462,
                        1892,  1329,  1393,  1748,  1304,  1763,  1634,  1764,   862,
                        842,   843,   844,   845,  1765,  1862,  1349,   931,   874,
                        874,  1255,  1255,  1862,  1601,   873,  1360,  1361,  1863,
                        1338,  1561,  1562,  1360,  1244,  1088,  1581,  1460,  1459,
                        1087,  1661,  1499,  1094,  1093,  1790,  1254,  1360,  1473,
                        1791,  1012,  1488,  1171,  1368,  1288,  1359,  1359,  1414,
                        884,  1801,   885,  1872,  1452,  1566,  1564,  1045,  1046,
                        1377,  1472,  1643,  1643,   988,  1068,  1207,  1552,  1552,
                        987,  1287,  1059,  1060,  1683,  1777,  1310,  1310,  1543,
                        1757,   867,  1114,   925,   937,  1106,   925,   936,  1106,
                        1835,  1425,  1424,  1258,  1580,  1641,  1199,  1199,   850,
                        1282,  1631,   976,  1907,  1231,  1671,   929,  1543,  1544,
                        979,  1347,  1347,  1816,  1816,  1838,  1018,   996,  1877,
                        1596,  1877,  1595,  1476,   941,  1829,  1829,  1008,  1008,
                        1232,  1233,  1007,  1890,  1524,  1543,  1415,   968,  1287,
                        1911,   857,  1062,  1191,  1191,  1921,  1233,  1481,  1507,
                        1508,  1882,  1641,  1543,  1585,  1586,  1670,  1671,   856,
                        857,  1760,  1043,  1044,  1007,  1008,   996,  1795,  1839,
                        1382,  1925,  1925,  1380,  2103,  2167,  2167,  2168,  2180,
                        1943,  1988,  2038,  1979,  1979,  1938,  2437,  1945,  2316,
                        1981,  1986,  2346,  2174,  2445,  2445,  2193,  1993,  2223,
                        2305,  2215,  2396,  2010,  2396,  2204,  2243,  2286,  2131,
                        2394,  2077,  2394,  1964,  2050,  2076,  2386,  1965,  2378,
                        2379,  2077,  2376,  2076,  1965,  2379,  1964,  2213,  2414,
                        2414,  1964,  2212,  2354,  2212,  2368,  2368,  2213,  2355,
                        2200,  2213,  2204,  2213,  2415,  2212,  2176,  2174,  2200,
                        2354,  1965,  2405,  2175,  2044,  2355,  2076,  2225,  2077,
                        2076,  2043,  2354,  2355,  2373,  2309,  2264,  2224,  2370,
                        1964,  2077,  2300,  2446,  2204,  2145,  2027,  2238,  1995,
                        2372,  2188,  2304,  2335,  2335,  2145,  2380,  2013,  2002,
                        1983,  2012,  1967,  1967,  2227,  1973,  2261,  1973,  2266,
                        2001,  2233,  2259,  2231,  2056,  2388,  2056,  2145,  2067,
                        2045,  2045,  2228,  2228,  1942,  2072,  2020,  1932,  1952,
                        1952,  2306,  2020,  2226,  1941,  1983,  2249,  2188,  2246,
                        2233,  2365,  2231,  2304,  2145,  2034,  2072,  2164,  2237,
                        2067,  2238,  2185,  2246,  2231,  2003,  2010,  2164,  2145,
                        2234,  2072,  2291,  2001,  2172,  2205,  2184,  2002,  2256,
                        2125,  1982,  2199,  2257,  2001,  2255,  2411,  2411,  2005,
                        2260,  2261,  2237,  2155,  2052,  2186,  2002,  2185,  2256,
                        1984,  2279,  1980,  1980,  2052,  2006,  2179,  1929,  2435,
                        1966,  2073,  2019,  2064,  2209,  2138,  2210,  2073,  2126,
                        2090,  1928,  2433,  2055,  2054,  1971,  2214,  2397,  2397,
                        2215,  2215,  2091,  2054,  2094,  2206,  2208,  1971,  2214,
                        2041,  1933,  2041,  2041,  1931,  1929,  2082,  1929,  2089,
                        2061,  2133,  1935,  2054,  1928,  2201,  2209,  2124,  1928,
                        2057,  2207,  2201,  2195,  2074,  1987,  1987,  2061,  2090,
                        2334,  2085,  2220,  2276,  2065,  2090,  2091,  2187,  2123,
                        2142,  2089,  1988,  1989,  2014,  2015,  2031,  1993,  2049,
                        2406,  2406,  2014,  2403,  2403,  2014,  2282,  2168,  2015,
                        2402,  2402,  2031,  2014,  2026,  2025,  2015,  2030,  1985,
                        2328,  2413,  2413,  2254,  2178,  2178,  1991,  1959,  2008,
                        2029,  2008,  2330,  2008,  2009,  2162,  2162,  2004,  2004,
                        1990,  2277,  2166,  2016,  2009,  1985,  2245,  2331,  1968,
                        2254,  2253,  2182,  2009,  2139,  2008,  2191,  2253,  2029,
                        2086,  2007,  2016,  2321,  2322,  2007,  1985,  2177,  2182,
                        1991,  2165,  2009,  2152,  1938,  1970,  1970,  2177,  2244,
                        2143,  2424,  2424,  2000,  2393,  2421,  2423,  2197,  2440,
                        2229,  2347,  2301,  2313,  2267,  2217,  2320,  2319,  2058,
                        1936,  2221,  1951,  1950,  1936,  2229,  1951,  1951,  2281,
                        2197,  2023,  2023,  2197,  1950,  2070,  1962,  2071,  1996,
                        1962,  2301,  2289,  2440,  2424,  2170,  1945,  1946,  1944,
                        2170,  2171,  2140,  1957,  2337,  2336,  2339,  2340,  2338,
                        1946,  1944,  1946,  1945,  1978,  2248,  2140,  2160,  1974,
                        2060,  1974,  1974,  1974,  2051,  1974,  1974,  1974,  2051,
                        2161,  2062,  2062,  2062,  2339,  2340,  2218,  2252,  2251,
                        2252,  2407,  2407,  2400,  2400,  2409,  2409,  2251,  2084,
                        2251,  2114,  2252,  2113,  2251,  2108,  2107,  2119,  2115,
                        2251,  2252,  2251,  2110,  1977,  2401,  2401,  2287,  2032,
                        2081,  2104,  2107,  2252,  2399,  2399,  2251,  2101,  1972,
                        2217,  2105,  2150,  2149,  2216,  2216,  2217,  2058,  2445,
                        2444,  1955,  1955,  2038,  2037,  2120,  2116,  2116,  2243,
                        2445,  2033,  2113,  1976,  1977,  1969,  2129,  1977,  2111,
                        2412,  2412,  1976,  1976,  1976,  1977,  2102,  1969,  2418,
                        1981,  1997,  2159,  2429,  2154,  2241,  2430,  2153,  1997,
                        2153,  2284,  2087,  2425,  2299,  2048,  1934,  1934,  1961,
                        2048,  2154,  2299,  2075,  1956,  1956,  2329,  2302,  2153,
                        2192,  1934,  2420,  2719,  2699,  2629,  2475,  2457,  2658,
                        2720,  2720,  2509,  2615,  2519,  2611,  2520,  2548,  2721,
                        2722,  2524,  2469,  2586,  2498,  2599,  2468,  2488,  2706,
                        2706,  2476,  2484,  2481,  2674,  2688,  2719,  2550,  2631,
                        2609,  2647,  2553,  2527,  2458,  2528,  2466,  2604,  2647,
                        2608,  2578,  2528,  2527,  2451,  2451,  2464,  2571,  2582,
                        2528,  2451,  2455,  2536,  2543,  2652,  2545,  2630,  2527,
                        2578,  2529,  2672,  2642,  2463,  2571,  2608,  2716,  2474,
                        2571,  2465,  2624,  2466,  2465,  2652,  2641,  2527,  2528,
                        2451,  2528,  2550,  2466,  2469,  2626,  2702,  2487,  2701,
                        2702,  2701,  2683,  2456,  2679,  2566,  2510,  2566,  2627,
                        2521,  2569,  2525,  2461,  2521,  2676,  2685,  2639,  2683,
                        2697,  2473,  2703,  2682,  2519,  2577,  2644,  2518,  2508,
                        2521,  2532,  2679,  2684,  2700,  2559,  2520,  2519,  2679,
                        2560,  2679,  2701,  2558,  2577,  2712,  2713,  2679,  2680,
                        2682,  2623,  2472,  2692,  2682,  2664,  2520,  2653,  2663,
                        2660,  2720,  2589,  2721,  2662,  2482,  2659,  2668,  2563,
                        2690,  2661,  2721,  2573,  2603,  2467,  2663,  2564,  2483,
                        2522,  2522,  2610,  2470,  2502,  2503,  2538,  2498,  2579,
                        2579,  2696,  2505,  2513,  2671,  2717,  2717,  2499,  2498,
                        2502,  2503,  2502,  2503,  2707,  2707,  2693,  2718,  2681,
                        2665,  2622,  2635,  2512,  2567,  2479,  2666,  2499,  2503,
                        2575,  2673,  2616,  2570,  2565,  2500,  2500,  2628,  2471,
                        2591,  2636,  2498,  2499,  2501,  2517,  2448,  2501,  2597,
                        2621,  2669,  2640,  2540,  2636,  2590,  2618,  2504,  2541,
                        2718,  2667,  2503,  2537,  2535,  2535,  2657,  2717,  2503,
                        2711,  2710,  2499,  2617,  2452,  2595,  2705,  2705,  2594,
                        2595,  2592,  2592,  2594,  2568,  2568,  2449,  2549,  2592,
                        2593,  2524,  2562,  2598,  2606,  2551,  2594,  2593,  2596,
                        2507,  2595,  2605,  2576,  2551,  2594,  2595,  2607,  2490,
                        2453,  2531,  2511,  2489,  2649,  2547,  2490,  2489,  2485,
                        2546,  2544,  2523,  2655,  2489,  2654,  2542,  2691,  2714,
                        2649,  2651,  2650,  2715,  2494,  2634,  2586,  2619,  2459,
                        2492,  2625,  2491,  2492,  2554,  2556,  2574,  2572,  2493,
                        824,   824,  1429,  1871,  1523,  2732,  2732,  1524,     4,
                        6,     8,    10,    11,    15,    17,    19,    21,    28,    36,
                        39,    42,    44,    47,    50,    52,    54,    56,    58,
                        61,    63,    71,    73,    75,    77,    79,    83,    85,
                        87,    89,    91,    93,    95,    97,    99,   101,   103,
                        105,   107,   110,   112,   115,   117,   119,   121,   124,
                        126,   128,   133,   137,   140,   143,   145,   147,   149,
                        152,   154,   156,   158,   161,   163,   164,   168,   170,
                        172,   175,   177,   183,   185,   187,   189,   192,   194,
                        196,   198,   200,   202,   204,   206,   212,   220,   227,
                        242,   248,   257,   264,   281,   283,   285,   297,   310,
                        318,   325,   334,   342,   344,   354,   355,   361,   375,
                        379,   385,   393,   404,   406,   421,   429,   436,   438,
                        441,   457,   464,   475,   485,   487,   489,   494,   507,
                        509,   515,   517,   520,   524,   530,   532,   536,   538,
                        541,   547,   551,   556,   574,   577,   587,   593,   613,
                        617,   664,   685,   696,   714,   730,   740,   758,   768,
                        778,   807,   846,   849,   851,   852,   854,   856,   860,
                        863,   865,   868,   870,   872,   875,   877,   882,   884,
                        890,   894,   896,   899,   900,   901,   902,   904,   906,
                        910,   916,   918,   920,   922,   926,   928,   930,   932,
                        934,   938,   940,   942,   945,   947,   953,   955,   959,
                        961,   963,   965,   967,   969,   971,   973,   975,   977,
                        979,   981,   985,   987,   989,   993,   995,   997,   999,
                        1002,  1003,  1005,  1007,  1009,  1011,  1013,  1014,  1015,
                        1018,  1024,  1025,  1035,  1041,  1043,  1045,  1047,  1049,
                        1051,  1057,  1059,  1061,  1064,  1067,  1069,  1071,  1073,
                        1075,  1077,  1079,  1081,  1085,  1087,  1091,  1095,  1101,
                        1103,  1107,  1109,  1111,  1113,  1114,  1115,  1116,  1118,
                        1122,  1132,  1134,  1137,  1138,  1143,  1144,  1148,  1150,
                        1155,  1164,  1166,  1170,  1172,  1177,  1179,  1181,  1185,
                        1187,  1194,  1201,  1204,  1206,  1208,  1210,  1212,  1214,
                        1218,  1220,  1222,  1224,  1230,  1232,  1234,  1236,  1240,
                        1242,  1244,  1246,  1248,  1254,  1256,  1258,  1262,  1264,
                        1266,  1268,  1269,  1276,  1278,  1282,  1285,  1287,  1289,
                        1291,  1293,  1295,  1297,  1299,  1301,  1303,  1305,  1307,
                        1311,  1313,  1315,  1319,  1323,  1329,  1331,  1333,  1335,
                        1337,  1342,  1344,  1347,  1349,  1351,  1352,  1353,  1354,
                        1356,  1360,  1362,  1364,  1366,  1368,  1374,  1375,  1376,
                        1377,  1379,  1381,  1383,  1385,  1387,  1389,  1391,  1393,
                        1397,  1402,  1408,  1410,  1412,  1414,  1416,  1418,  1420,
                        1422,  1426,  1428,  1432,  1437,  1439,  1443,  1447,  1449,
                        1451,  1453,  1455,  1457,  1459,  1461,  1463,  1467,  1471,
                        1473,  1475,  1477,  1479,  1481,  1483,  1485,  1487,  1489,
                        1492,  1494,  1496,  1498,  1500,  1502,  1504,  1509,  1511,
                        1517,  1519,  1521,  1523,  1525,  1527,  1535,  1537,  1541,
                        1543,  1545,  1547,  1553,  1555,  1557,  1561,  1563,  1567,
                        1569,  1571,  1573,  1575,  1577,  1579,  1581,  1583,  1585,
                        1587,  1589,  1591,  1593,  1597,  1601,  1603,  1605,  1607,
                        1611,  1613,  1617,  1619,  1623,  1625,  1627,  1629,  1630,
                        1631,  1632,  1634,  1636,  1640,  1644,  1646,  1649,  1653,
                        1659,  1662,  1664,  1668,  1670,  1672,  1673,  1674,  1675,
                        1677,  1679,  1682,  1684,  1686,  1688,  1690,  1694,  1696,
                        1698,  1702,  1705,  1707,  1709,  1713,  1715,  1717,  1719,
                        1725,  1727,  1729,  1731,  1734,  1736,  1738,  1744,  1746,
                        1748,  1750,  1752,  1754,  1756,  1760,  1762,  1764,  1766,
                        1768,  1770,  1772,  1774,  1776,  1780,  1786,  1788,  1790,
                        1793,  1794,  1796,  1801,  1803,  1805,  1807,  1809,  1811,
                        1817,  1819,  1821,  1823,  1825,  1827,  1830,  1832,  1834,
                        1836,  1837,  1838,  1839,  1843,  1845,  1847,  1849,  1851,
                        1853,  1858,  1860,  1862,  1864,  1866,  1872,  1878,  1880,
                        1882,  1884,  1886,  1888,  1890,  1892,  1894,  1896,  1900,
                        1902,  1904,  1906,  1910,  1913,  1915,  1917,  1921,  1923,
                        1926,  1928,  1936,  1941,  1944,  1947,  1950,  1957,  1962,
                        1964,  1974,  1976,  1982,  1991,  1993,  1997,  1999,  2001,
                        2005,  2008,  2011,  2014,  2017,  2023,  2032,  2037,  2041,
                        2043,  2046,  2059,  2063,  2068,  2070,  2073,  2076,  2090,
                        2101,  2105,  2107,  2110,  2112,  2119,  2121,  2127,  2129,
                        2131,  2134,  2137,  2143,  2145,  2147,  2153,  2157,  2170,
                        2175,  2177,  2183,  2197,  2206,  2209,  2212,  2214,  2216,
                        2218,  2221,  2224,  2229,  2233,  2235,  2237,  2246,  2251,
                        2253,  2257,  2260,  2262,  2273,  2277,  2287,  2289,  2293,
                        2311,  2313,  2319,  2341,  2343,  2347,  2350,  2354,  2358,
                        2360,  2363,  2366,  2368,  2370,  2373,  2375,  2378,  2384,
                        2386,  2389,  2421,  2423,  2440,  2442,  2444,  2453,  2465,
                        2470,  2489,  2491,  2498,  2500,  2502,  2505,  2519,  2520,
                        2527,  2533,  2587,  2592,  2594,  2608,  2614,  2629,  2650,
                        2660,  2662,  2679,  2697,  2701,  2717,  2720,  2733])

    # [np.flatnonzero(bus_nrs == bus)[0] for bus in from_bus_nr]
    br_f = np.array([np.flatnonzero(bus_nrs == bus)[0] for bus in from_bus_nr])
    br_t = np.array([np.flatnonzero(bus_nrs == bus)[0] for bus in to_bus_nr])

    if rm_line is not None: # Remove lines from the system
        R_br = np.delete(R_br, rm_line)
        X_br = np.delete(X_br, rm_line)
        G_br = np.delete(G_br, rm_line)
        B_br = np.delete(B_br, rm_line)

        br_f = np.delete(br_f, rm_line)
        br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    branch['Tap'] = tr_ratio

    return bus, branch, gen



def small_test(Vxd=None,Xd=None,X1=None,X2=None,Pg=None,ZLD=None,Test=None,Sbus=None,Xbr=None):
    # Xd = Xdpu*Zbase1*1j
    # X1 = 5.0j
    # X2 = 12.5j
    # Pg = 24.871
    # Pld = 38.5
    if Test is not None:
        # Vxd = 39.48
        n_bus = 4
        # n_br = 3
        Sbase = 50
        Vbase = 20
        Zbase = Vbase**2/Sbase
        
        bus_type = np.array([2, 2, 1, 3])
        bus_nr = np.arange(n_bus)
        Vm = np.array([abs(Vxd), 20.00, 18.53, 20.00])/Vbase
        Va = np.zeros(n_bus)
        # if Sbus is not None:
        Pgen = np.array([Pg, 0.0, 0.0, 0])/Sbase
        Qgen = np.array([0.0, 0.0, 0.0,0])/Sbase
       
        Pld = np.zeros(n_bus)
        Qld = np.zeros(n_bus)
        
        # Sbus[Sbus.real<0].real
        Gsh = np.array([0.0, 0.0, 1/ZLD, 0.0])*Zbase
        # Gsh = np.zeros(n_bus)
        Bsh = np.zeros(n_bus)
        # Gsh[Sbus.real<0] = np.zeros(n_bus)
        # Bsh[Sbus.real<0] = np.zeros(n_bus)
        # else:
        #     Pgen = np.array([24.871, 0.0, 0.0, 13.63])/Sbase
        #     Qgen = np.array([0.0, 0.0, 0.0,0])/Sbase
        
        #     Pld = np.array([0.0, 0.0, 0.0, 0.0])/Sbase
        #     Qld = np.array([0.0, 0.0, 0.0, 0.0])/Sbase
    
        #     Gsh = np.array([0.0, 0.0, 1/8.921, 0.0])*Zbase
        #     Bsh = np.zeros(n_bus)
    
        R_br = np.array([0.00, 0.00, 0.00])
        X_br = np.array([abs(Xd), abs(X1), abs(X2)])/Zbase
        G_br = np.array([0.00, 0.00, 0.00])
        B_br = np.array([0.00, 0.00, 0.00])
    
        br_f = np.array([1, 2, 3])-1
        br_t = np.array([2, 3, 4])-1
        
    elif Vxd is not None:
        # Vxd = 39.48
        n_bus = 4
        # n_br = 3
        Sbase = 50
        Vbase = 20
        Zbase = Vbase**2/Sbase
        
        bus_type = np.array([2, 2, 1, 3])
        bus_nr = np.arange(n_bus)
        Vm = np.array([abs(Vxd), 20.00, 18.53, 20.00])/Vbase
        Va = np.zeros(n_bus)
        if Sbus is not None:
            Pgen = np.zeros(n_bus)
            Qgen = np.zeros(n_bus)
            Pgen[Sbus.real>0] = Sbus[Sbus.real>0].real
            Qgen[Sbus.real>0] = Sbus[Sbus.real>0].imag
        
            Pld = np.zeros(n_bus)
            Qld = np.zeros(n_bus)
            
            # Sbus[Sbus.real<0].real
            Gsh = np.array([0.0, 0.0, 1/8.921, 0.0])*Zbase
            # Gsh = np.zeros(n_bus)
            Bsh = np.zeros(n_bus)
            # Gsh[Sbus.real<0] = np.zeros(n_bus)
            # Bsh[Sbus.real<0] = np.zeros(n_bus)
        else:
            Pgen = np.array([24.871, 0.0, 0.0, 13.63])/Sbase
            Qgen = np.array([0.0, 0.0, 0.0,0])/Sbase
        
            Pld = np.array([0.0, 0.0, 0.0, 0.0])/Sbase
            Qld = np.array([0.0, 0.0, 0.0, 0.0])/Sbase
    
            Gsh = np.array([0.0, 0.0, 1/8.921, 0.0])*Zbase
            Bsh = np.zeros(n_bus)
    
        R_br = np.array([0.00, 0.00, 0.00])
        X_br = np.array([20.32, 5, 12.5])/Zbase
        G_br = np.array([0.00, 0.00, 0.00])
        B_br = np.array([0.00, 0.00, 0.00])
    
        br_f = np.array([1, 2, 3])-1
        br_t = np.array([2, 3, 4])-1
    else:
        n_bus = 3
        # n_br = 3
        Sbase = 50
        Vbase = 20
        Zbase = Vbase**2/Sbase
        
        bus_type = np.array([2, 1, 3])
        bus_nr = np.arange(n_bus)
        Vm = np.array([20.00, 18.53, 20.00])/Vbase
        Va = np.zeros(n_bus)
        Pgen = np.array([24.871, 0.0, 13.63])/Sbase
        Qgen = np.array([0.0, 0.0, 0.0])/Sbase
    
        Pld = np.array([0.0, 38.5, 0.0])/Sbase
        Qld = np.array([0.0, 0.0, 0.0])/Sbase
    
        Gsh = np.array([0.0, 0.0, 0.0])*Zbase
        Bsh = np.zeros(n_bus)
    
        R_br = np.array([0.00, 0.00])
        X_br = np.array([5, 12.5])/Zbase
        G_br = np.array([0.00, 0.00])
        B_br = np.array([0.00, 0.00])
    
        br_f = np.array([1, 2])-1
        br_t = np.array([2, 3])-1

    # if rm_line is not None: # Remove lines from the system
    #     R_br = np.delete(R_br, rm_line)
    #     X_br = np.delete(X_br, rm_line)
    #     G_br = np.delete(G_br, rm_line)
    #     B_br = np.delete(B_br, rm_line)

    #     br_f = np.delete(br_f, rm_line)
    #     br_t = np.delete(br_t, rm_line)

    n_br = len(br_f)
    
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen


def two_bus_test(X=5,PV=False):

    n_bus = 2
    # n_br = 3
    Sbase = 100
    Vbase = 20
    Zbase = Vbase**2/Sbase
    
    if PV:
        bus_type = np.array([3, 2])
    else:
        bus_type = np.array([3, 1])
    bus_nr = np.arange(n_bus)
    Vm = np.array([20.00, 20.00])/Vbase
    Va = np.zeros(n_bus)
    Pgen = np.array([0, 0.0])/Sbase
    Qgen = np.array([0.0, 0.0])/Sbase

    Pld = np.array([0.0, 10])/Sbase
    Qld = np.array([0.0, 0.0])/Sbase

    Gsh = np.array([0.0, 0.0])*Zbase
    Bsh = np.zeros(n_bus)

    R_br = np.array([0.00, 0.00])
    X_br = np.array([X])/Zbase
    G_br = np.array([0.00, 0.00])
    B_br = np.array([0.00, 0.00])

    br_f = np.array([1])-1
    br_t = np.array([2])-1

    
    n_br = len(br_f)
    
    # TODO multiple generators on the same bus
    genbus = np.flatnonzero(np.logical_or.reduce((Pgen != 0,Qgen !=0, bus_type==3)))
    n_gen = len(genbus)

    init_dicts(n_bus, n_br, n_gen)

    bus['Nr'] = bus_nr
    bus['Type'] = bus_type
    bus['Pld'] = Pgen-Pld
    bus['Qld'] = Qgen-Qld
    bus['Bsh'] = Bsh
    bus['Gsh'] = Gsh
    bus['Vm'] = Vm
    bus['Va'] = Va

    gen['bus'] = genbus
    gen['P'] = Pgen[genbus]
    gen['Q'] = Qgen[genbus]
    gen['V'] = Vm[genbus]
    gen['MVA'] += 100
    gen['Pmax'] += 1
    gen['Qmax'] += 1

    branch['From'] = br_f
    branch['To'] = br_t
    branch['R'] = R_br
    branch['X'] = X_br
    branch['B'] = B_br
    branch['rating'] = np.full(n_br,max(Pld)*1.5) # TODO add propper line rating
    # branch['Tap'] = tr_ratio

    return bus, branch, gen
