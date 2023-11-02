
import numpy as np
import scipy.linalg as la
import copy


def PowerFlowNewton(Ybus, Sbus, V0, ref, pv_index, pq_index, max_iter, err_tol):
    success = False #Initialization of status flag and iteration counter
    n = 0
    V = V0
    # print(' iteration maximum P & Q mismatch (pu)')
    # print(' --------- ---------------------------')
    # Determine mismatch between initial guess and and specified value for P and Q
    F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)
    # Check if the desired tolerance is reached
    success, normF = CheckTolerance(F, err_tol)
    # Start the Newton iteration loop

    while (not success) and (n < max_iter) :
        # print(n,normF)
        n += 1 # Update counter
        # Compute derivatives and generate the Jacobian matrix
        J_dS_dVm , J_dS_dTheta = generate_Derivatives(Ybus, V)
        J = generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index)
        # Compute the update step
        try:
            dx = la.solve(J, F)
            # Update voltages and check if tolerance is now reached
            V = Update_Voltages(dx, V, pv_index, pq_index)
            F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)
            success, normF = CheckTolerance(F, err_tol)
        except la.LinAlgWarning:
            print('Ill conditioned matrix')
            break
        except:
            print('Singular matrix')
            break
        finally:
            pass
    # print(n,normF)
    # if not success: #print out message concerning wether the power flow converged or not
    #     print('No Convergence !!!\n Stopped after %d iterations without solution...' %(n, ))

    if any(np.isnan(V)):
        print('No Convergence !!!\n NAN in voltage vector')
        success = False
    # else :
    #     print('The Newton Rapson Power Flow Converged in %d iterations!' %(n, ))
    return V, success, n

def calculate_F(Ybus, Sbus, V, pv_index, pq_index):
    Delta_S = Sbus-V*(Ybus.dot(V)).conj() # This function calculates the mismatch between the specified values of P and Q (In term of S)

    # We only use the above function for PQ and PV buses.
    Delta_P = np.real(Delta_S)
    Delta_Q = np.imag(Delta_S)

    F = np.concatenate((Delta_P[pv_index], Delta_P[pq_index], Delta_Q[pq_index]), axis = 0)

    return F

def CheckTolerance(F, err_tol):
    normF = la.norm(F,np.inf)

    if normF > err_tol:
        success = False
        # print('Not Success')
    else:
        success = True
    #     print('Success')
    # print('Highest error %.3f' %normF)
    return success, normF

def generate_Derivatives(Ybus, V):
    V = V.reshape(-1, 1)

    J_dS_dVm = (V.conj()*Ybus*(V/np.abs(V)).T).conj()
    np.fill_diagonal(J_dS_dVm, J_dS_dVm.diagonal() + (V/np.abs(V)*(Ybus@V).conj()).squeeze())
    J_dS_dTheta = -1j*V*(Ybus*V.T).conj()
    np.fill_diagonal(J_dS_dTheta, J_dS_dTheta.diagonal() + (1j*V*(Ybus@V).conj()).squeeze())

    return J_dS_dVm, J_dS_dTheta

def generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index):
    pvpq_ind=np.append(pv_index, pq_index)

    J_11 = np.real(J_dS_dTheta[np.ix_(pvpq_ind, pvpq_ind)])
    J_12 = np.real(J_dS_dVm[np.ix_(pvpq_ind, pq_index)])
    J_21 = np.imag(J_dS_dTheta[np.ix_(pq_index, pvpq_ind)])
    J_22 = np.imag(J_dS_dVm[np.ix_(pq_index, pq_index)])

    J = np.block([[J_11,J_12],[J_21,J_22]])

    return J

def Update_Voltages(dx, V, pv_index, pq_index):
    N1 = 0; N2 = len(pv_index) # dx[N1:N2]-ang. on the pv buses
    N3 = N2; N4 = N3 + len(pq_index) # dx[N3:N4]-ang. on the pq buses
    N5 = N4; N6 = N5 + len(pq_index) # dx[N5:N6]-mag. on the pq buses
    Theta = np.angle(V); Vm = np.absolute(V)
    if len(pv_index)>0:
        Theta[pv_index] += dx[N1:N2]
    if len(pq_index)>0:
        Theta[pq_index] += dx[N3:N4]
        Vm[pq_index] += dx[N5:N6]
    V = Vm * np.exp(1j*Theta)

    return V

def DisplayResults(V, Ybus, Y_from, Y_to, br_f, br_t):
    S_to = V[br_t]*(Y_to.dot(V)).conj()
    S_from = V[br_f]*(Y_from.dot(V)).conj()
    S_inj = V*(Ybus.dot(V)).conj()

    dash = '=' * 60
    print(dash)
    print('|{:^58s}|'.format('Bus results'))
    print(dash)
    print('{:^6s} {:^17s} {:^17s} {:^17s}'.format('Bus', 'Voltage', 'Generation','Load'))
    print('{:^6s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('#', 'Mag(pu)', 'Ang(deg)','P(pu)', 'Q(pu)','P(pu)', 'Q(pu)'))
    print('{:^6s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('-'*6,'-'*8, '-'*8,'-'*8, '-'*8,'-'*8, '-'*8))

    for i in range(0 ,len(V)):
        if np.real(S_inj[i]) > 0:
            print('{:^6d} {:^8.3f} {:^8.3f} {:^8.3f} {:^8.3f} {:^8s} {:^9s}'.format(i+1, np.abs(V[i]), np.rad2deg(np.angle(V[i])), np.real(S_inj[i]), np.imag(S_inj[i]),'-','-'))
        else:
            print('{:^6d} {:^8.3f} {:^8.3f} {:^8s} {:^8s} {:^8.3f} {:^9.3f}'.format(i+1, np.abs(V[i]), np.rad2deg(np.angle(V[i])),'-','-', -np.real(S_inj[i]), -np.imag(S_inj[i])))

    print(dash)
    print('|{:^58s}|'.format('Branch Flow'))
    print(dash)
    print('{:^6s} {:<6s} {:<6s} {:^19s} {:^19s}'.format('Branch', 'From','To','From bus Injection', 'To bus Injection'))
    print('{:^6s} {:<6s} {:<6s} {:^9s} {:^9s} {:^9s} {:^9s}'.format('#','Bus','Bus','P(pu)', 'Q(pu)','P(pu)', 'Q(pu)'))


    print('{:^5s} {:^5s} {:^5s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('-'*6,'-'*6, '-'*6,'-'*9, '-'*9,'-'*9, '-'*9))

    for i in range(0 ,len(br_f)):
        print('{:^6d} {:^6d} {:^6d} {:^9.3f} {:^9.3f} {:^9.3f} {:^9.3f}'.format(i+1, br_f[i]+1, br_t[i]+1, -np.real(S_from[i]), -np.imag(S_from[i]), -np.real(S_to[i]), -np.imag(S_to[i])))


def PowerFlowFD(Ybus , Sbus , V0 , ref , pv_index , pq_index, max_iter, err_tol):

    success = False
    n = 0
    V = V0
    Va = np.angle(V)
    Vm = abs(V)

    dQ = pq_index
    dV = pq_index

    dP = np.concatenate((pv_index, pq_index))
    dTheta = np.concatenate((pv_index, pq_index))

    mis = (Sbus - V * np.conj(Ybus @ V)) / Vm

    # (Sbus - V * np.conj(Ybus @ V)) / Vm
    P = np.real(mis[dP])
    Q = np.imag(mis[dQ])

    normP = la.norm(P,np.inf)
    if len(pq_index) > 0:
        normQ = la.norm(Q,np.inf)
    else:
        normQ = 0
        
    if normP < err_tol and normQ < err_tol:
        success = True

    Bp = np.imag(Ybus[dP, :][:, dTheta])
    Bpp = np.imag(Ybus[dQ, :][:, dV])


    mis = (Sbus - V * np.conj(Ybus @ V))
    P = np.real(mis[dP])/ Vm[dP]
    Q = np.imag(mis[dQ])/ Vm[dQ]

    Bp_inv = la.inv(Bp)
    
    if len(pq_index) > 0:
        Bpp_inv = la.inv(Bpp)
        
    while not success and n < max_iter:

        n += 1

        dVa = -Bp_inv@P


        Va[dP]  = Va[dP]  + dVa
        V = Vm * np.exp(1j * Va)

        mis = (Sbus - V * np.conj(Ybus @ V)) / Vm
        P = np.real(mis[dP])
        Q = np.imag(mis[dQ])

        normP = la.norm(P, np.inf)
        if len(pq_index) > 0:
            normQ = la.norm(Q, np.inf)

        if normP < err_tol and normQ < err_tol:
            success = True
            break

        if len(pq_index) > 0:
            dVm = -Bpp_inv@Q
        else:
            dVm = 0

        Vm[dQ] = Vm[dQ] + dVm
        V = Vm * np.exp(1j * Va)

        mis = -(V * np.conj(Ybus @ V) - Sbus) / Vm
        P = np.real(mis[dP])
        Q = np.imag(mis[dQ])

        normP = la.norm(P, np.inf)
        if len(pq_index) > 0:
            normQ = la.norm(Q, np.inf)

        if normP < err_tol and normQ < err_tol:
            success = True
            break

    return V, success, n

def PowerFlowDC(B, Bf, Pbusinj, Pfinj, Gsh, Va0, Sbus, ref, pv_index, pq_index):
    # Gsh = bus['Gsh']
    # Va0 = bus['Va']
    Pbus = np.real(Sbus) - Pbusinj - Gsh

    Va = copy.copy(Va0)
    pvpq = np.concatenate([pv_index,pq_index])
    
    try:
        Va[pvpq] = la.solve(B[:, pvpq][pvpq,:],(Pbus[pvpq] - B[pvpq, ref] * Va0[ref]))
    except:
        return np.ones(len(Va0)), 0, np.zeros(len(Bf))
    PF = (Bf @ Va + Pfinj)# * 100
    PT = -PF
    V = np.ones(len(Va0))* np.exp(1j * Va)

    PG = (B[ref, :] @ Va - Pbus[ref])

    success = 1

    return V, success, PF

