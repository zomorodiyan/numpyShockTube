# Translation of G. Mellema's Roe Solver

import math

def Roe_solve(dr, dt, gamma, vol, state, flux, meshr, icntl):
    """ dr      spatial step
        dt      time step
        gamma   adiabatic index
        vol     volume factor for 3-D problem
        state   (rho, rho*u, e) -- input
        flux    flux at cell boundaries -- output
        meshr   number of interior points
        icntl   diagnostic -- bad if != 0
    """
    tiny = 1e-30
    sbpar1 = 2.0
    sbpar2 = 2.0

    # allocate temporary arrays
    fludif = [ [0.0] * 3 for i in range(meshr + 2) ]
    rsumr = [ 0.0 for i in range(meshr + 2) ]
    utilde = [ 0.0 for i in range(meshr + 2) ]
    htilde = [ 0.0 for i in range(meshr + 2) ]
    absvt = [ 0.0 for i in range(meshr + 2) ]
    uvdif = [ 0.0 for i in range(meshr + 2) ]
    ssc = [ 0.0 for i in range(meshr + 2) ]
    vsc = [ 0.0 for i in range(meshr + 2) ]
    a = [ [0.0] * 3 for i in range(meshr + 2) ]
    ac1 = [ [0.0] * 3 for i in range(meshr + 2) ]
    ac2 = [ [0.0] * 3 for i in range(meshr + 2) ]
    w = [ [0.0] * 4 for i in range(meshr + 2) ]
    eiglam = [ [0.0] * 3 for i in range(meshr + 2) ]
    sgn = [ [0.0] * 3 for i in range(meshr + 2) ]
    fluxc = [ [0.0] * 3 for i in range(meshr + 2) ]
    fluxl = [ [0.0] * 3 for i in range(meshr + 2) ]
    fluxr = [ [0.0] * 3 for i in range(meshr + 2) ]
    ptest = [ 0.0 for i in range(meshr + 2) ]
    isb = [ [0] * 3 for i in range(meshr + 2) ]

    # initialize control variable to 0
    icntl[0] = 0

    # find parameter vector w
    for i in range(meshr + 2):
        w[i][0] = math.sqrt(vol[i] * state[i][0])
        w[i][1] = w[i][0] * state[i][1] / state[i][0]
        w[i][3] = (gamma - 1) * (state[i][2] - 0.5 * state[i][1]
                  * state[i][1] / state[i][0])
        w[i][2] = w[i][0] * (state[i][2] + w[i][3]) / state[i][0]

    # calculate the fluxes at the cell center
    for i in range(meshr + 2):
        fluxc[i][0] = w[i][0] * w[i][1]
        fluxc[i][1] = w[i][1] * w[i][1] + vol[i] * w[i][3]
        fluxc[i][2] = w[i][1] * w[i][2]

    # calculate the fluxes at the cell walls
    # assuming constant primitive variables
    for n in range(3):
        for i in range(1, meshr + 2):
            fluxl[i][n] = fluxc[i - 1][n]
            fluxr[i][n] = fluxc[i][n]

    # calculate the flux differences at the cell walls
    for n in range(3):
        for i in range(1, meshr + 2):
            fludif[i][n] = fluxr[i][n] - fluxl[i][n]

    # calculate the tilded state variables = mean values at the interfaces
    for i in range(1, meshr + 2):
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
            icntl[0] += 1

    # calculate the eigenvalues and projection coefficients for each eigenvector
    for i in range(1, meshr + 2):
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
        for i in range(1, meshr + 2):
            a[i][n] /= eiglam[i][n] + tiny

    # calculate the first order projection coefficients ac1
    for n in range(3):
        for i in range(1, meshr + 2):
            ac1[i][n] = - sgn[i][n] * a[i][n] * eiglam[i][n]

    # apply the 'superbee' flux correction to made 2nd order projection
    # coefficients ac2
    for n in range(3):
        ac2[1][n] = ac1[1][n]
        ac2[meshr + 1][n] = ac1[meshr + 1][n]

    dtdx = dt / dr
    for n in range(3):
        for i in range(2, meshr + 1):
            isb[i][n] = i - int(sgn[i][n])
            ac2[i][n] = (ac1[i][n] + eiglam[i][n] *
                        ((max(0.0, min(sbpar1 * a[isb[i][n]][n], max(a[i][n],
                        min(a[isb[i][n]][n], sbpar2 * a[i][n])))) +
                        min(0.0, max(sbpar1 * a[isb[i][n]][n], min(a[i][n],
                        max(a[isb[i][n]][n], sbpar2 * a[i][n])))) ) *
                        (sgn[i][n] - dtdx * eiglam[i][n])))

    # calculate the final fluxes
    for i in range(1, meshr + 2):
        flux[i][0] = 0.5 * (fluxl[i][0] + fluxr[i][0] + ac2[i][0]
                     + ac2[i][1] + ac2[i][2])
        flux[i][1] = 0.5 * (fluxl[i][1] + fluxr[i][1] +
                     eiglam[i][0] * ac2[i][0] + eiglam[i][1] * ac2[i][1] +
                     eiglam[i][2] * ac2[i][2])
        flux[i][2] = 0.5 * (fluxl[i][2] + fluxr[i][2] +
                     (htilde[i] - utilde[i] * vsc[i]) * ac2[i][0] +
                     absvt[i] * ac2[i][1] +
                     (htilde[i] + utilde[i] * vsc[i]) * ac2[i][2])

    # calculate test variable for negative pressure check
    for i in range(1, meshr + 1):
        ptest[i] = (dr * vol[i] * state[i][1] +
                   dt * (flux[i][1] - flux[i + 1][1]))
        ptest[i] = (- ptest[i] * ptest[i] + 2 * (dr * vol[i] * state[i][0] +
                   dt * (flux[i][0] - flux[i + 1][0])) * (dr * vol[i] *
                   state[i][2] + dt * (flux[i][2] - flux[i + 1][2])))

    # check for negative pressure/internal energy and set fluxes
    # left and right to first order if detected
    for i in range(1, meshr + 1):
        if (ptest[i] <= 0.0 or (dr * vol[i] * state[i][0] + dt * (flux[i][0]
                                - flux[i + 1][0])) <= 0.0):

            flux[i][0] = 0.5 * (fluxl[i][0] + fluxr[i][0] +
                ac1[i][0] + ac1[i][1] + ac1[i][2])
            flux[i][1] = 0.5 * (fluxl[i][1] + fluxr[i][1] +
                eiglam[i][0] * ac1[i][0] + eiglam[i][1] * ac1[i][1] +
                eiglam[i][2] * ac1[i][2])
            flux[i][2] = 0.5 * (fluxl[i][2] + fluxr[i][2] +
                (htilde[i]-utilde[i] * vsc[i]) * ac1[i][0] +
                absvt[i] * ac1[i][1] +
                (htilde[i] + utilde[i] * vsc[i]) * ac1[i][2])
            flux[i + 1][0] = 0.5 * (fluxl[i + 1][0] + fluxr[i + 1][0] +
                 ac1[i + 1][0] + ac1[i + 1][1] + ac1[i + 1][2])
            flux[i + 1][1] = 0.5 * (fluxl[i + 1][1] + fluxr[i + 1][1] +
                 eiglam[i + 1][0] * ac1[i + 1][0] + eiglam[i + 1][1] *
                 ac1[i + 1][1] + eiglam[i + 1][2] * ac1[i + 1][2])
            flux[i + 1][2] = 0.5 * (fluxl[i + 1][2] + fluxr[i + 1][2] +
                 (htilde[i + 1] - utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][0]
                 + absvt[i + 1] * ac1[i + 1][1] +
                 (htilde[i + 1] + utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][2])

            # Check if it helped, set control variable if not

            ptest[i] = (dr * vol[i] * state[i][1] +
                       dt * (flux[i][1] - flux[i + 1][1]))
            ptest[i] = (2.0 * (dr * vol[i] * state[i][0]
                + dt * (flux[i][0]-flux[i + 1][0])) * (dr * vol[i] *
                state[i][2] + dt * (flux[i][2] - flux[i + 1][2]))
                - ptest[i] * ptest[i])
            if (ptest[i] <= 0.0 or (dr * vol[i] * state[i][0] +
                    dt * (flux[i][0] - flux[i + 1][0])) <= 0.0):
                icntl[0] += 1
