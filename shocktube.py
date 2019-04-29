# Euler equations for Sod's shocktube problem

import math
# from tools import Roe_solve
import numpy as np

L = 1.0                     # length of shock tube
gamma = 1.4                 # ratio of specific heats
N = 200                     # number of grid points

CFL = 0.9                   # Courant-Friedrichs-Lewy number
nu = 0.0                    # artificial viscosity coefficient


def initialize(L, N):           # 48N
    """
    creates variables U, U_new, F, vol with shape of (N, 3) and 0.0 value
    defines e by this equation:... and use left_state and right_state and
    e to initialize U[j, :] = (rho[j], rho*v[j], e[j]) and vol[j] = 1 and
    tau = CFL * h / dc_max()

    , h tau globaly
    """
    U = [[0.0] * 3 for j in range(N)]  # 3N
    h = L / float(N - 1)

    for j in range(N):  # 18N
        # left_state
        rho = 1.0
        P = 1.0
        v = 0.0
        if j > int(N / 2):
            # right_state
            rho = 0.125
            P = 0.1
            # v = 0.0
        e = P / (gamma - 1) + rho * v**2 / 2
        U[j][0] = rho
        U[j][1] = rho * v
        U[j][2] = e
        # vol[j] = 1.0

    tau = CFL * h / c_max(U)  # c_max() = 20N
    return U, h, tau


def out_file(U, plot, file_name, t):
    # write solution in plot files and print averages
    file = open(str(plot) + "-" + file_name, "w")
    rho_avg = 0.0
    v_avg = 0.0
    e_avg = 0.0
    P_avg = 0.0
    N = len(U[:][0])
    for j in range(N):
        rho = U[j][0]
        v = U[j][1] / U[j][0]
        e = U[j][2]
        P = (U[j][2] - U[j][1] * U[j][1] / U[j][0] / 2) * (gamma - 1)
        rho_avg += rho
        v_avg += v
        e_avg += e
        P_avg += P
        file.write(str(j) + '\t' + str(rho) + '\t' + str(v) + '\t'
                   + str(e) + '\t' + str(P) + '\n')
    if rho_avg != 0.0:
        rho_avg /= N
    if v_avg != 0.0:
        v_avg /= N
    if e_avg != 0.0:
        e_avg /= N
    if P_avg != 0.0:
        P_avg /= N
    print(" ", t, '\t', rho_avg, '\t', v_avg, '\t', e_avg, '\t', P_avg)
    file.close()


def boundary_conditions(U):  # 0N
    # reflection boundary conditions at the tube ends
    print("method, boundary_conditions is running")
    N = len(U[:][0])
    U[0][0] = U[1][0]
    U[0][1] = -U[1][1]
    U[0][2] = U[1][2]
    U[N - 1][0] = U[N - 2][0]
    U[N - 1][1] = -U[N - 2][1]
    U[N - 1][2] = U[N - 2][2]
    return U


def c_max(U):   # 22N
    """
    param: ...
    rtype = float
    returns max(c + abs(v)) in U[:]
    """
    print("method, c_max is running")
    v_max = 0.0
    N = len(U[:][0])
    for j in range(N):
        if U[j][0] != 0.0:
            rho = U[j][0]
            v = U[j][1] / rho
            P = (U[j][2] - rho * v**2 / 2) * (gamma - 1)
            c = math.sqrt(gamma * abs(P) / rho)
            if v_max < c + abs(v):
                v_max = c + abs(v)
    return v_max


def Lax_Wendroff_step(h, tau, U, gamma=1.4):  # 92N
    print("method, Lax_Wendroff_step is running")
    # global F, U, U_new
    N = len(U[:][0])
    U_new = [[0.0] * 3 for j in range(N)]  # 3N
    F = [[0.0] * 3 for j in range(N)]  # 3N
    # compute flux F from U
    for j in range(N):  # 19N
        rho = U[j][0]
        m = U[j][1]
        e = U[j][2]
        P = (gamma - 1) * (e - m**2 / rho / 2)
        F[j][0] = m
        F[j][1] = m**2 / rho + P
        F[j][2] = m / rho * (e + P)

    # half step
    for j in range(1, N - 1):   # 30N + boundary_conditions() = 30N
        for i in range(3):
            U_new[j][i] = ((U[j + 1][i] + U[j][i]) / 2 -
                           tau / 2 / h * (F[j + 1][i] - F[j][i]))
    boundary_conditions(U_new)

    # compute flux at half steps
    for j in range(N):  # 19N
        rho = U_new[j][0]
        m = U_new[j][1]
        e = U_new[j][2]
        P = (gamma - 1) * (e - m**2 / rho / 2)
        F[j][0] = m
        F[j][1] = m**2 / rho + P
        F[j][2] = m / rho * (e + P)

    # step using half step flux
    for j in range(1, N - 1):   # 21N
        for i in range(3):
            U_new[j][i] = U[j][i] - tau / h * (F[j][i] - F[j - 1][i])

    # update U from U_new
    for j in range(1, N - 1):   # 3N
        for i in range(3):
            U[j][i] = U_new[j][i]
    return U


def Roe_step(h, tau, U, gamma=1.4):  # Roe_solve() + 15N
    # compute fluxes at cell boundaries
    # icntl = [0] #!!! (I removed it from Roe_solve inputs, maybe I merge Roe_step and Roe_solve)
    """ h      spatial step
        tau      time step
        gamma   adiabatic index
        vol     volume factor for 3-D problem
        U   (rho, rho*u, e) -- input
        F    flux at cell boundaries -- output
        N   number of points
        icntl   diagnostic -- bad if != 0
    """
    print("method, Roe_step is running")
    # allocate temporary arrays
    tiny = 1e-30
    sbpar1 = 2.0
    sbpar2 = 2.0
    N = len(U[:][0])
    vol = np.ones((N))
    F = [[0.0] * 3 for j in range(N)]  # 3N
    fludif = [ [0.0] * 3 for i in range(N) ]
    rsumr = [ 0.0 for i in range(N) ]
    utilde = [ 0.0 for i in range(N) ]
    htilde = [ 0.0 for i in range(N) ]
    absvt = [ 0.0 for i in range(N) ]
    uvdif = [ 0.0 for i in range(N) ]
    ssc = [ 0.0 for i in range(N) ]
    vsc = [ 0.0 for i in range(N) ]
    a = [ [0.0] * 3 for i in range(N) ]
    ac1 = [ [0.0] * 3 for i in range(N) ]
    ac2 = [ [0.0] * 3 for i in range(N) ]
    w = [ [0.0] * 4 for i in range(N) ]
    eiglam = [ [0.0] * 3 for i in range(N) ]
    sgn = [ [0.0] * 3 for i in range(N) ]
    Fc = [ [0.0] * 3 for i in range(N) ]
    Fl = [ [0.0] * 3 for i in range(N) ]
    Fr = [ [0.0] * 3 for i in range(N) ]
    ptest = [ 0.0 for i in range(N) ]
    isb = [ [0] * 3 for i in range(N) ]

    # initialize control variable to 0
    icntl = 0

    # find parameter vector w
    for i in range(N):
        w[i][0] = math.sqrt(vol[i] * U[i][0])
        w[i][1] = w[i][0] * U[i][1] / U[i][0]
        w[i][3] = (gamma - 1) * (U[i][2] - 0.5 * U[i][1]
                  * U[i][1] / U[i][0])
        w[i][2] = w[i][0] * (U[i][2] + w[i][3]) / U[i][0]

    # calculate the fluxes at the cell center
    for i in range(N):
        Fc[i][0] = w[i][0] * w[i][1]
        Fc[i][1] = w[i][1] * w[i][1] + vol[i] * w[i][3]
        Fc[i][2] = w[i][1] * w[i][2]

    # calculate the fluxes at the cell walls
    # assuming constant primitive variables
    for n in range(3):
        for i in range(1, N):
            Fl[i][n] = Fc[i - 1][n]
            Fr[i][n] = Fc[i][n]

    # calculate the flux differences at the cell walls
    for n in range(3):
        for i in range(1, N):
            fludif[i][n] = Fr[i][n] - Fl[i][n]

    # calculate the tilded U variables = mean values at the interfaces
    for i in range(1, N):
        rsumr[i] = 1 / (w[i - 1][0] + w[i][0])

        utilde[i] = (w[i - 1][1] + w[i][1]) * rsumr[i]
        htilde[i] = (w[i - 1][2] + w[i][2]) * rsumr[i]

        absvt[i] = 0.5 * utilde[i] * utilde[i]
        uvdif[i] = utilde[i] * fludif[i][1]

        ssc[i] = (gamma - 1) * (htilde[i] - absvt[i])
        if ssc[i] > 0.0:
            vsc[i] = math.sqrt(ssc[i])
        else:
            vsc[i] = math.sqrt(abs(ssc[i]))
            icntl += 1

    # calculate the eigenvalues and projection coefficients for each eigenvector
    for i in range(1, N):
        eiglam[i][0] = utilde[i] - vsc[i]
        eiglam[i][1] = utilde[i]
        eiglam[i][2] = utilde[i] + vsc[i]
        for n in range(3):
            if eiglam[i][n] < 0.0:
                sgn[i][n] = -1
            else:
                sgn[i][n] = 1
        a[i][0] = 0.5 * ((gamma - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
                  - uvdif[i]) - vsc[i] * (fludif[i][1] - utilde[i]
                  * fludif[i][0])) / ssc[i]
        a[i][1] = (gamma - 1) * ((htilde[i] - 2 * absvt[i]) * fludif[i][0]
                  + uvdif[i] - fludif[i][2]) / ssc[i]
        a[i][2] = 0.5 * ((gamma - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
                  - uvdif[i]) + vsc[i] * (fludif[i][1] - utilde[i]
                  * fludif[i][0])) / ssc[i]

    # divide the projection coefficients by the wave speeds
    # to evade expansion correction
    for n in range(3):
        for i in range(1, N):
            a[i][n] /= eiglam[i][n] + tiny

    # calculate the first order projection coefficients ac1
    for n in range(3):
        for i in range(1, N):
            ac1[i][n] = - sgn[i][n] * a[i][n] * eiglam[i][n]

    # apply the 'superbee' flux correction to made 2nd order projection
    # coefficients ac2
    for n in range(3):
        ac2[1][n] = ac1[1][n]
        ac2[N - 1][n] = ac1[N - 1][n]

    dtdx = tau / h
    for n in range(3):
        for i in range(2, N -1):
            isb[i][n] = i - int(sgn[i][n])
            ac2[i][n] = (ac1[i][n] + eiglam[i][n] *
                        ((max(0.0, min(sbpar1 * a[isb[i][n]][n], max(a[i][n],
                        min(a[isb[i][n]][n], sbpar2 * a[i][n])))) +
                        min(0.0, max(sbpar1 * a[isb[i][n]][n], min(a[i][n],
                        max(a[isb[i][n]][n], sbpar2 * a[i][n])))) ) *
                        (sgn[i][n] - dtdx * eiglam[i][n])))

    # calculate the final fluxes
    for i in range(1, N):
        F[i][0] = 0.5 * (Fl[i][0] + Fr[i][0] + ac2[i][0]
                     + ac2[i][1] + ac2[i][2])
        F[i][1] = 0.5 * (Fl[i][1] + Fr[i][1] +
                     eiglam[i][0] * ac2[i][0] + eiglam[i][1] * ac2[i][1] +
                     eiglam[i][2] * ac2[i][2])
        F[i][2] = 0.5 * (Fl[i][2] + Fr[i][2] +
                     (htilde[i] - utilde[i] * vsc[i]) * ac2[i][0] +
                     absvt[i] * ac2[i][1] +
                     (htilde[i] + utilde[i] * vsc[i]) * ac2[i][2])

    # calculate test variable for negative pressure check
    for i in range(1, N - 1):
        ptest[i] = (h * vol[i] * U[i][1] +
                   tau * (F[i][1] - F[i + 1][1]))
        ptest[i] = (- ptest[i] * ptest[i] + 2 * (h * vol[i] * U[i][0] +
                   tau * (F[i][0] - F[i + 1][0])) * (h * vol[i] *
                   U[i][2] + tau * (F[i][2] - F[i + 1][2])))

    # check for negative pressure/internal energy and set fluxes
    # left and right to first order if detected
    for i in range(1, N - 1):
        if (ptest[i] <= 0.0 or (h * vol[i] * U[i][0] + tau * (F[i][0]
                                - F[i + 1][0])) <= 0.0):

            F[i][0] = 0.5 * (Fl[i][0] + Fr[i][0] +
                ac1[i][0] + ac1[i][1] + ac1[i][2])
            F[i][1] = 0.5 * (Fl[i][1] + Fr[i][1] +
                eiglam[i][0] * ac1[i][0] + eiglam[i][1] * ac1[i][1] +
                eiglam[i][2] * ac1[i][2])
            F[i][2] = 0.5 * (Fl[i][2] + Fr[i][2] +
                (htilde[i]-utilde[i] * vsc[i]) * ac1[i][0] +
                absvt[i] * ac1[i][1] +
                (htilde[i] + utilde[i] * vsc[i]) * ac1[i][2])
            F[i + 1][0] = 0.5 * (Fl[i + 1][0] + Fr[i + 1][0] +
                 ac1[i + 1][0] + ac1[i + 1][1] + ac1[i + 1][2])
            F[i + 1][1] = 0.5 * (Fl[i + 1][1] + Fr[i + 1][1] +
                 eiglam[i + 1][0] * ac1[i + 1][0] + eiglam[i + 1][1] *
                 ac1[i + 1][1] + eiglam[i + 1][2] * ac1[i + 1][2])
            F[i + 1][2] = 0.5 * (Fl[i + 1][2] + Fr[i + 1][2] +
                 (htilde[i + 1] - utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][0]
                 + absvt[i + 1] * ac1[i + 1][1] +
                 (htilde[i + 1] + utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][2])

            # Check if it helped, set control variable if not

            ptest[i] = (h * vol[i] * U[i][1] +
                       tau * (F[i][1] - F[i + 1][1]))
            ptest[i] = (2.0 * (h * vol[i] * U[i][0]
                + tau * (F[i][0]-F[i + 1][0])) * (h * vol[i] *
                U[i][2] + tau * (F[i][2] - F[i + 1][2]))
                - ptest[i] * ptest[i])
            if (ptest[i] <= 0.0 or (h * vol[i] * U[i][0] + tau * (F[i][0] - F[i + 1][0])) <= 0.0):
                icntl += 1


    # update U
    for j in range(1, N - 1):   # 15N
        for i in range(3):
            U[j][i] -= tau / h * (F[j + 1][i] - F[j][i])
    return U


def Lapidus_viscosity(h, tau, U):    # 33N
    print("method, lapidus_viscosity is running")
    # store Delta_U values in newU
    N = len(U[:][0])
    U_new = [[0.0] * 3 for j in range(N)]  # 3N
    for j in range(1, N):   # 9N
        for i in range(3):
            U_new[j][i] = U[j][i] - U[j - 1][i]

    # multiply Delta_U by |Delta_U|
    for j in range(1, N):   # 6N
        for i in range(3):
            U_new[j][i] *= abs(U_new[j][i])

    # add artificial viscosity
    for j in range(2, N):   # 18N
        for i in range(3):
            U[j][i] += nu * tau / h * (U_new[j][i] - U_new[j - 1][i])
    return U


def solve(step_algorithm, t_max, file_name, plots=5):
    U, h, tau = initialize(L, N)
    t = 0.0
    step = 0
    plot = 0
    print(" Time t\t\trho_avg\t\tu_avg\t\te_avg\t\tP_avg")
    while True:
        out_file(U, plot, file_name, t)
        plot += 1
        if plot > plots:
            print(" Solutions in files 0-..", plots, "-" + file_name)
            break

        while t < t_max * plot / float(plots):
            U = boundary_conditions(U)
            tau = CFL * h / c_max(U)
            U = step_algorithm(h, tau, U)
            U = Lapidus_viscosity(h, tau, U)
            t += tau
            step += 1

print(" Sod's Shocktube Problem using various algorithms:")
print(" 1. A two-step Lax-Wendroff algorithm")
print(" 2. Mellema's Roe solver")
print(" N = ", N, ", CFL Number = ", CFL, ", nu = ", nu)
print()
print(" Lax-Wendroff Algorithm")
solve(Lax_Wendroff_step, 1.0, "lax.data")
print()
print(" Roe Solver Algorithm")
solve(Roe_step, 1.0, "roe.data")
