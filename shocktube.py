# Euler equations for Sod's shocktube problem

import math
from tools import Roe_solve

L = 1.0                     # length of shock tube
gamma = 1.4                 # ratio of specific heats
N = 200                     # number of grid points

CFL = 0.9                   # Courant-Friedrichs-Lewy number
nu = 0.0                    # artificial viscosity coefficient


def c_max():                # 22N
    """
    param: ...
    rtype = float
    returns max(c + abs(v)) in U[:]
    """
    v_max = 0.0
    for j in range(N):
        if U[j][0] != 0.0:
            rho = U[j][0]
            v = U[j][1] / rho
            P = (U[j][2] - rho * v**2 / 2) * (gamma - 1)
            c = math.sqrt(gamma * abs(P) / rho)
            if v_max < c + abs(v):
                v_max = c + abs(v)
    return v_max


def initialize():           # 48N
    """
    creates variables U, U_new, F, vol with shape of (N, 3) and 0.0 value
    defines e by this equation:... and use left_state and right_state and
    e to initialize U[j, :] = (rho[j], rho*v[j], e[j]) and vol[j] = 1 and 
    tau = CFL * h / dc_max()

    , h tau globaly

    """
    global U, U_new, F, vol, h, tau

    U = [[0.0] * 3 for j in range(N)]  # 3N
    U_new = [[0.0] * 3 for j in range(N)]  # 3N
    F = [[0.0] * 3 for j in range(N)]  # 3N
    vol = [0.0 for j in range(N)]  # N

    h = L / float(N - 1)  
    for j in range(N):  # 18N
        #left_state
        rho = 1.0
        P = 1.0
        v = 0.0
        if j > int(N / 2):
            #right_state
            rho = 0.125
            P = 0.1
            #v = 0.0
        e = P / (gamma - 1) + rho * v**2 / 2
        U[j][0] = rho
        U[j][1] = rho * v
        U[j][2] = e
        vol[j] = 1.0

    tau = CFL * h / c_max()  # c_max() = 20N


def boundary_conditions(U):  # 0N
    # reflection boundary conditions at the tube ends
    U[0][0] = U[1][0]
    U[0][1] = -U[1][1]
    U[0][2] = U[1][2]
    U[N - 1][0] = U[N - 2][0]
    U[N - 1][1] = -U[N - 2][1]
    U[N - 1][2] = U[N - 2][2]


def Lax_Wendroff_step():    #92N
    global F, U, U_new

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


def Roe_step(): # Roe_solve() + 15N
    global U

    # compute fluxes at cell boundaries
    icntl = [0]
    Roe_solve(h, tau, gamma, vol, U, F, N - 2, icntl)

    # update U
    for j in range(1, N - 1):   # 15N
        for i in range(3):
            U[j][i] -= tau / h * (F[j + 1][i] - F[j][i])


def Lapidus_viscosity():    # 33N
    global U, U_new

    # store Delta_U values in newU
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


def solve(step_algorithm, t_max, file_name, plots=5):
    global tau

    initialize()
    t = 0.0
    step = 0
    plot = 0

    print(" Time t\t\trho_avg\t\tu_avg\t\te_avg\t\tP_avg")
    while True:

        # write solution in plot files and print averages
        file = open(str(plot) + "-" + file_name, "w")
        rho_avg = 0.0
        v_avg = 0.0
        e_avg = 0.0
        P_avg = 0.0
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

        plot += 1
        if plot > plots:
            print(" Solutions in files 0-..", plots, "-" + file_name)
            break

        while t < t_max * plot / float(plots):
            boundary_conditions(U)
            tau = CFL * h / c_max()
            step_algorithm()
            Lapidus_viscosity()
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
