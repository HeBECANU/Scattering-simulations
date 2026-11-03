import numpy as np
from scipy.integrate import solve_ivp
from uncertainties import ufloat
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import json
import os

#CONSTANTS - in AMU MICROMETRE MICROSECOND UNITS
hbar = 0.06350632
m_F = 3.016  #mass of 3He
m_B = 4.0026 #mass of 4He
m_r = m_F*m_B/(m_F+m_B)  #reduced mass
a_bb = 7.512e-3 #between m_j = 1 and m_j = 1
a_bf = 30e-3 #30e-3 or -9e-3, up for debate #between m_j = 1 and m_f = 3/2
kB = 8314.46 #boltzmann constant
g = 4*np.pi*hbar**2*a_bb/m_B #boson self-interaction
h = 2*np.pi*hbar**2*a_bf/m_r #boson_fermion interaction
a = h/g 
wavelength = 1.083 #wavelength of helium transition
p = hbar*2*np.pi/wavelength #momentum of light 'kick'
vrec = p/m_B 
crossSectionbf = 4*np.pi*a_bf**2 #bose-fermi cross-section
crossSectionbb = 8*np.pi*a_bb**2 #cross-section for bose-bose collisions (m_j=1)
tight = (96e-6, 650e-6, 650e-6)
weak = (75e-6, 84e-6, 84e-6)
ultraweak = (15e-6, 25e-6, 25e-6)

TRAPS = {"ultraweak": ultraweak, "weak": weak, "tight": tight,}

dullPurple = tuple(int("BF40BF"[i:i+2], 16) /255 for i in (0, 2, 4)) 
medGray = tuple(int("999999"[i:i+2], 16)/255  for i in (0, 2, 4))
lightGray = tuple(int("eeeeee"[i:i+2], 16)/255  for i in (0, 2, 4))
gold = tuple(int("ffbb6f"[i:i+2], 16)/255 for i in (0, 2, 4))

TRAP_COLORS = {"ultraweak": gold, "weak": medGray, "tight": dullPurple,}



def sampleDensity(fx, fy, fz, T, mu_f, mu_b, x, y, z, tolerance = 0.001):
    '''
    Inputs:
    mu_f, mu_b, chemical potentials of fermions and bosons
    fx, fy, fz, trapping frequencies in MHz
    T, temperature in K

    Outputs:
    n_f, n_b, n_nc, arrays of shape (len(x_vals), len(y_vals), len(z_vals))
    '''
    wx, wy, wz = 2*np.pi*fx, 2*np.pi*fy, 2*np.pi*fz
    wbar = (wx*wy*wz)**(1/3)

    V = 0.5*m_B*(wx**2*x**2+wy**2*y**2+wz**2*z**2)

    if T == 0:
        #fermi-sphere treatment
        #no thermal bosons
        densityTerm = np.where(mu_b > V, mu_f+(a-1)*V-a*mu_b, mu_f - V)
        n_f = np.where(densityTerm > 0, 1/(6*np.pi**2) * (2*m_F*densityTerm / hbar**2)**(3/2), 0)
        n_b = np.where(mu_b - V - h*n_f > 0, (mu_b - V - h*n_f)/g, 0)
        return n_f, n_b, np.zeros_like(n_b)
    else:
        Veff = np.where(mu_b > V, (1-a)*V+a*mu_b, V)
        n_f = fermion_density(Veff, mu_f, T, 600, 75)
        n_b = np.where(mu_b - V - h*n_f > 0, (mu_b - V - h*n_f)/g, 0)
        Veff_boson = V + 2*g*n_b + h*n_f
        n_nc = thermal_density(Veff_boson, mu_b, T, 600, 75)
        error1, error2, error3 = 2*tolerance, 2*tolerance, 2*tolerance
        #loop until densities converge
        while error1 > tolerance or error2 > tolerance or error3 > tolerance:
            n_fOld = n_f
            n_ncOld = n_nc
            n_bOld = n_b
            Veff_fermion = np.where(mu_b > V, (1-a)*V + a*mu_b + h*n_nc, V + h*n_nc)
            n_f = fermion_density(Veff_fermion, mu_f, T, 600, 75)
            n_b = np.where(mu_b - V - h*n_f - 2*g*n_nc > 0, (mu_b- V- h*n_f - 2*g*n_nc)/g, 0)
            Veff_boson = V + 2*g*n_b + 2*g*n_nc + h*n_f
            n_nc = thermal_density(Veff_boson, mu_b, T, 600, 75)
            def safe_rel_error(new, old):
                denom = np.sum(new)
                return np.sum(np.abs(new - old)) / denom if denom != 0 else 2*tolerance

            error1 = safe_rel_error(n_f, n_fOld)
            error2 = safe_rel_error(n_nc, n_ncOld)
            error3 = safe_rel_error(n_b, n_bOld)            
            print("Errors", error1, error2, error3)
        if x.ndim == 3:
            thermalbosons = round(integrateDensity(n_nc, x[:, 0, 0], y[0, :, 0], z[0, 0, :]))
            condensatebosons = round(integrateDensity(n_b, x[:, 0, 0], y[0, :, 0], z[0, 0, :]))
            print(f"Thermal Boson Number: {thermalbosons}, Condensate Boson Number: {condensatebosons}")
        return n_f, n_b, n_nc

def fermion_density(V_eff_grid, mu_f, T, N_p, chunk):
    E_max = np.max(mu_f - V_eff_grid)
    p_max = np.sqrt(2 * m_F * (E_max + 6 * kB * T))

    p = np.linspace(0, p_max, N_p)
    integral = np.zeros_like(V_eff_grid)

    for i in range(0, N_p - 1, chunk):
        p_chunk = p[i : i + chunk + 1]
        # Reshape p to broadcast against whatever shape V_eff_grid is
        p_ = p_chunk.reshape((-1,) + (1,) * V_eff_grid.ndim)
        V_ = V_eff_grid[np.newaxis, ...]

        exponent = (p_**2 / (2 * m_F) + V_ - mu_f) / (kB * T)
        np.clip(exponent, -500, 500, out=exponent)
        integrand = p_**2 / (np.exp(exponent) + 1.0)

        integral += np.trapz(integrand, p_chunk, axis=0)

    return integral / (2*np.pi**2*hbar**3)

def thermal_density(V_eff_grid, mu_B, T, N_p, chunk):
    E_max = np.max(mu_B - V_eff_grid)
    p_max = np.sqrt(2 * m_F * (E_max + 6 * kB * T))

    p = np.linspace(0, p_max, N_p)
    integral = np.zeros_like(V_eff_grid)

    for i in range(0, N_p - 1, chunk):
        p_chunk = p[i : i + chunk + 1]
        # Reshape p to broadcast against whatever shape V_eff_grid is
        p_ = p_chunk.reshape((-1,) + (1,) * V_eff_grid.ndim)
        V_ = V_eff_grid[np.newaxis, ...]

        exponent = (p_**2 / (2 * m_B) + V_ - mu_B) / (kB * T)
        np.clip(exponent, -500, 500, out=exponent)
        integrand = p_**2 / (np.exp(exponent) - 1.0)

        integral += np.trapz(integrand, p_chunk, axis=0)

    return integral / (2*np.pi**2*hbar**3)

def generatePositiveArrays(mu_b, mu_f, fx, fy, fz, numX, numY):
    '''
    Generates the positive octant up to the edge of the atomic cloud
    '''
    wx = 2*np.pi*fx
    wy = 2*np.pi*fy
    wz = 2*np.pi*fz

    rx_F = np.sqrt(2*mu_f/(m_B*wx**2))
    ry_F = np.sqrt(2*mu_f/(m_B*wy**2))
    rz_F = np.sqrt(2*mu_f/(m_B*wz**2))
    rx_B = np.sqrt(2*mu_b/(m_B*wx**2))
    ry_B = np.sqrt(2*mu_b/(m_B*wy**2))
    rz_B = np.sqrt(2*mu_b/(m_B*wz**2))
    rx = np.max([rx_F, rx_B])
    ry = np.max([ry_F, ry_B])
    rz = np.max([rz_F, rz_B])
    x_vals = np.linspace(0, rx, numX)
    y_vals = np.linspace(0, ry, numY)
    z_vals = np.linspace(0, rz, numY)
    return x_vals, y_vals, z_vals

def generateFullArrays(mu_b, mu_f, fx, fy, fz, numX, numY):
    '''
    Generates the positive octant up to the edge of the atomic cloud
    '''
    wx = 2*np.pi*fx
    wy = 2*np.pi*fy
    wz = 2*np.pi*fz

    rx_F = np.sqrt(2*mu_f/(m_B*wx**2))
    ry_F = np.sqrt(2*mu_f/(m_B*wy**2))
    rz_F = np.sqrt(2*mu_f/(m_B*wz**2))
    rx_B = np.sqrt(2*mu_b/(m_B*wx**2))
    ry_B = np.sqrt(2*mu_b/(m_B*wy**2))
    rz_B = np.sqrt(2*mu_b/(m_B*wz**2))
    rx = np.max([rx_F, rx_B])
    ry = np.max([ry_F, ry_B])
    rz = np.max([rz_F, rz_B])
    x_vals = np.linspace(-rx, rx, numX)
    y_vals = np.linspace(-ry, ry, numY)
    z_vals = np.linspace(-rz, rz, numY)
    return x_vals, y_vals, z_vals


def analyticBEC(fx, fy, fz, mu_b, x, y, z):
    V = m_B*2*np.pi**2*(fx**2*x**2+fy**2*y**2+fz**2*z**2)
    n_b = np.where(mu_b > V, 1/g*(mu_b - V), 0)
    return n_b

def integrateDensity(density, x_vals, y_vals, z_vals):
    '''
    Integrates the density function over the x y and z arrays given
    '''
    return np.trapz(np.trapz(np.trapz(density, z_vals, axis=2), y_vals, axis=1), x_vals)

def evaluateNormalisedInnerProduct(n1, n2, x_vals, y_vals, z_vals):
    return integrateDensity(n1*n2, x_vals, y_vals, z_vals)/np.sqrt(integrateDensity(n1**2, x_vals, y_vals, z_vals))/np.sqrt(integrateDensity(n2**2, x_vals, y_vals, z_vals))

def evaluateL2Norm(n1, n2, x_vals, y_vals, z_vals):
    return np.sqrt(integrateDensity((n1-n2)**2, x_vals, y_vals, z_vals))

def evaluateCriticalTemperature(fx, fy, fz, N_b, Nerr=None):
    if Nerr:
        N_b = ufloat(N_b, Nerr)
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    wm = 2*np.pi*(fx+fy+fz)/3
    T_c = hbar*wbar*0.94*N_b**(1/3)/kB
    T_c += (-0.73*wm/wbar*N_b**(-1/3)*T_c-1.33*a_bb*np.sqrt(m_B*wbar/hbar)*N_b**(1/6)*T_c)
    return T_c

def evaluateFermiTemperature(fx, fy, fz, N_f):
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    return (6*N_f)**(1/3)*hbar*wbar/kB

def castinDum(t, state, wx, wy, wz):
    bx, by, bz, dbx, dby, dbz = state
    d2bx = wx**2 / (bx * by * bz * bx) 
    d2by = wy**2 / (bx * by * bz * by) 
    d2bz = wz**2 / (bx * by * bz * bz) 
    return [dbx, dby, dbz, d2bx, d2by, d2bz]

def sampleExpandedBEC(mu_B, mu_F, fx, fy, fz, x, y, z, t, T):
    '''
    Takes in total energies and potential and samples expanded BEC profile based on the Krachmalnicoff equations
    for a particular time at any number of points
    '''


    state0 = [1, 1, 1, 0, 0, 0]
    t_max = float(np.max(t))
    sol = solve_ivp(castinDum, [0, t_max], state0, 
                    args=(2*np.pi*fx, 2*np.pi*fy, 2*np.pi*fz), 
                    dense_output=True)
    # evaluate scaling factors at each t
    b = sol.sol(t.ravel())  # shape (6, N_times)
    scaleX = b[0]
    scaleY = b[1]
    scaleZ = b[2]
    
    x = x/scaleX
    y = y/scaleY
    z = z/scaleZ
    _, n_B, _ = sampleDensity(fx, fy, fz, T, mu_F, mu_B, x, y, z)
    n_B = n_B  / scaleX / scaleY / scaleZ
    return  n_B

def generateExpandedArray(mu, fx, fy, fz, numX, numY, t):
    state0 = [1, 1, 1, 0, 0, 0]
    t_max = float(np.max(t))
    sol = solve_ivp(castinDum, [0, t_max], state0, 
                    args=(2*np.pi*fx, 2*np.pi*fy, 2*np.pi*fz), 
                    dense_output=True)
    # evaluate scaling factors at each t
    b = sol.sol(t.ravel())  # shape (6, N_times)
    scaleX = b[0]
    scaleY = b[1]
    scaleZ = b[2]
    x_vals, y_vals, z_vals = generateFullArrays(mu, 0, fx, fy, fz, numX, numY)
    x_vals, y_vals, z_vals = x_vals*scaleX, y_vals*scaleY, z_vals*scaleZ
    return x_vals, y_vals, z_vals

def evaluateEnergies(fx, fy, fz, T, N_f, N_b, alpha=0.2, tolerance=0.001,
                      cache_file="energy_cache.json"):

    # --- check cache for existing result ---
    def params_match(p, fx, fy, fz, T, N_f, N_b, rtol=1e-9):
        keys = ["fx", "fy", "fz", "T", "N_f", "N_b"]
        vals = [fx, fy, fz, T, N_f, N_b]
        return all(np.isclose(p[k], v, rtol=rtol, atol=0) for k, v in zip(keys, vals))

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = []
    else:
        cache = []

    for entry in cache:
        if params_match(entry, fx, fy, fz, T, N_f, N_b):
            print("Matching parameters found in cache, returning stored energies.")
            return entry["mu_f"], entry["mu_b"]

    print("Beginning energy evaluation for parameters:")
    print(f"Frequencies: ({fx*1e6}, {fy*1e6}, {fz*1e6}) Hz, Temperature: {T} K, Atom Numbers: {N_f} fermions, {N_b} bosons")
    #initialise guess for chemical potentials
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    mu_f = hbar*np.sqrt(m_B/m_F)*wbar*(6*N_f)**(1/3)
    mu_b = ((m_B*wbar**2/2)**(3/2)*15/(8*np.pi)*g*N_b)**(2/5)
    T_c = hbar*wbar*0.94*N_b**(1/3)/kB
    wm = 2*np.pi*(fx+fy+fz)/3
    T_c += (-0.73*wm/wbar*N_b**(-1/3)*T_c-1.33*a_bb*np.sqrt(m_B*wbar/hbar)*N_b**(1/6)*T_c)
    if T > T_c:
        print("Operating above the critical temperature - simulation failed.")
        return 0, 0
    print("Critical Temperature:", T_c)

    N_f_calc = 2*tolerance*N_f + N_f
    N_b_calc = 2*tolerance*N_b + N_b

    #loop until convergence
    while np.abs(N_f_calc - N_f)/N_f > tolerance or np.abs(N_b_calc - N_b)/N_b > tolerance:
        if N_f != 0:
            mu_f *= (N_f/N_f_calc)**alpha
        mu_b *= (N_b/N_b_calc)**alpha

        #sample densities
        x_vals, y_vals, z_vals = generatePositiveArrays(mu_b, mu_f, fx, fy, fz, 100, 100)
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing= 'ij')

        n_f, n_b, n_nc = sampleDensity(fx, fy, fz, T, mu_f, mu_b, x, y, z)

        #integrate over densities to get particle number
        N_f_calc = 8*integrateDensity(n_f, x_vals, y_vals, z_vals)
        N_b_calc = 8*integrateDensity(n_b+n_nc, x_vals, y_vals, z_vals)
        print(f"Boson Error: {np.abs(N_b_calc - N_b)/N_b}, Fermion Error: {np.abs(N_f_calc - N_f)/N_f}")

    print(f"Energy evaluated! Fermion Chemical Potential: {mu_f}, Boson Chemical Potential: {mu_b}")

    # --- store result in cache ---
    cache.append({
        "fx": fx, "fy": fy, "fz": fz,
        "T": T,
        "N_f": N_f, "N_b": N_b,
        "mu_f": mu_f, "mu_b": mu_b
    })
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    return mu_f, mu_b

def evaluateCollisions(mu_f, mu_b, fx, fy, fz, T, nf, tfree=1500):
    wx, wy, wz = 2*np.pi*fx, 2*np.pi*fy, 2*np.pi*fz
    rz_B = np.sqrt(2*mu_b/(m_B*wz**2))
    rz_F = np.sqrt(2*mu_f/(m_B*wz**2))
    maxT = 1.2*(rz_F+rz_B)*m_B/p #factor of 1.2 allows for the clouds to become a bit larger due to expansion
    t_array = np.linspace(0, maxT, 50)

    rx = np.sqrt(2*mu_f/(m_B*wx**2))
    ry = np.sqrt(2*mu_f/(m_B*wy**2))
    rz = np.sqrt(2*mu_f/(m_B*wz**2))

    # Build proper 3D grid axes
    N_grid = 200
    xs = np.linspace(-rx, rx, N_grid)
    ys = np.linspace(-ry, ry, N_grid)
    zs = np.linspace(-rz, rz, N_grid)

    # Evaluate density on the full 3D grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')  
    n_f, _, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, X, Y, Z) 
    collisionRates = np.zeros((len(t_array), 30))
    for i in range(30):
        print(f"\rEvaluating collisions {i}/30", end="", flush=True)
        collisionRates[:,i] = boseFermiCollision(xs, ys, zs, n_f, t_array, mu_b, mu_f, T, fx, fy, fz, nf, tfree= tfree)
    collisionRate = np.average(collisionRates, axis=1)
    collisions = np.trapezoid(collisionRates, t_array, axis=0)
    avg, err = np.average(collisions), np.std(collisions)    
    print()
    return collisionRate, t_array, avg, err

def boseBoseCollision(mu_b, fx, fy, fz, tfree, vkick = np.sqrt(2)*vrec, transfer=0.5, fraction = 1):
    v = vkick
    wz = 2*np.pi*fz
    rz_B = np.sqrt(2*mu_b/(m_B*wz**2))
    maxT = 1.2*2*rz_B/v 
    T = np.linspace(0, maxT, 50)
    collision_rate = np.zeros_like(T)
    for i in range(len(T)):
        # print("Timestep:", i+1)
        t = T[i]

        #First, we generate arrays for the timestep
        x_vals, y_vals, z_vals = generateExpandedArray(mu_b, fx, fy, fz, 80, 80, t+tfree)
        kicked_z = z_vals + v*t
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        #Now solve for the density profile. Only one is needed since they are identical just translated
        bec = fraction*sampleExpandedBEC(mu_b, 0, fx, fy, fz, X, Y, Z, t+tfree, 0)
        kickedbec = bec*transfer
        unkickedbec = bec*(1-transfer)

        #Now we need to interpolate the values of the arrays in a shared region
        shared_z = np.linspace(kicked_z[0], z_vals[-1], 80)
        interp1 = RegularGridInterpolator((x_vals, y_vals, z_vals), unkickedbec, bounds_error=False, fill_value=0)
        interp2 = RegularGridInterpolator((x_vals, y_vals, kicked_z), kickedbec, bounds_error=False, fill_value=0)

        pts = np.array(np.meshgrid(x_vals, y_vals, shared_z, indexing='ij')).reshape(3, -1).T
        bec1 = interp1(pts).reshape(80, 80, 80)
        bec2 = interp2(pts).reshape(80, 80, 80)


        collision_rate[i] = crossSectionbb*v*integrateDensity(unkickedbec*kickedbec, x_vals, y_vals, shared_z) 
    return collision_rate, T, np.trapezoid(collision_rate, T)

def boseFermiCollision(xs, ys, zs, n_f, t_array, mu_B, mu_F, T, fx, fy, fz, N_F, n_fermions_sample=2000, time_points=50, tfree = 1500):
    '''
    This function is used in the Bose-Fermi case, where we sample individual fermion positions and momenta and evaluate
    the likelihoods of each individual particle to collide
    '''    
    #Sample fermion position and velocities
    fp = sampleFermionPositions(xs, ys, zs, n_f, N_F, n_fermions_sample)
    fm = sampleFermionMomenta(mu_B, mu_F, T, fx, fy, fz, fp)
    velocities = fm / m_F

    #we want to simulate a free expansion for the first tfree microseconds
    #move fermions 
    positions = fp[:,None,:] + velocities[:,None,:] * tfree

    #Add velocity kick to fermions
    velocities[:,2] += np.sqrt(2)*vrec
    #Update position according to velocity
    positions = positions + velocities[:,None,:] * t_array[None,:,None]

    #Sample density at fermion position
    densities = sampleExpandedBEC(mu_B, mu_F, fx, fy, fz, positions[:,:,0], positions[:,:,1], positions[:,:,2], t_array[None,:]+tfree, 0)

    #Evaluate collision rate contribution from sum over densities
    collision_rate =  crossSectionbf* np.sqrt(2)* vrec * densities.sum(axis=0) 
    collision_rate *= N_F / n_fermions_sample
    return collision_rate

def sampleFermionPositions(xs, ys, zs, n_f, N_F, n_sample=5000, batch=10000):
    '''
    Take in 3D fermion density grid built from chemical potentials and randomly samples fermion positions on the grid
    '''
   
    p_max = (n_f / N_F).max() * 1.05
    # Rejection sampling over grid indices
    accepted_coords = []
    n_accepted = 0
    n_total = 0
    while n_accepted < n_sample:
        n_total += 1
        xi = np.random.randint(0, len(xs), size=batch)
        yi = np.random.randint(0, len(ys), size=batch)
        zi = np.random.randint(0, len(zs), size=batch)

        p_f = n_f[xi, yi, zi] / N_F   
        u = np.random.uniform(0, p_max, size=batch)
        mask = u < p_f

        # Convert accepted indices to physical coordinates
        coords = np.column_stack([xs[xi[mask]], ys[yi[mask]], zs[zi[mask]]])
        accepted_coords.append(coords)
        n_accepted += mask.sum()
    result = np.concatenate(accepted_coords)[:n_sample]

    return result  # shape (n_sample, 3)

def sampleFermionMomenta(mu_B, mu_F, T, fx, fy, fz, fp):
    wx, wy, wz = 2*np.pi*fx, 2*np.pi*fy, 2*np.pi*fz
    x, y, z = fp[:,0], fp[:,1], fp[:,2]
    
    V = 0.5*m_B*(wx**2*x**2 + wy**2*y**2 + wz**2*z**2)
    V_eff = np.where(mu_B > V, (1-a)*V + a*mu_B, V)
    E_local = mu_F - V_eff

    p_max = np.sqrt(2 * m_F * (np.maximum(E_local, 0) + 6*kB*T))
    k_max = p_max / hbar

    n = len(fp)
    k_accepted = np.zeros(n)
    accepted = np.zeros(n, dtype=bool)

    while not np.all(accepted):
        remaining = ~accepted
        k_candidates = np.random.uniform(0, k_max[remaining])
        E_k = hbar**2 * k_candidates**2 / (2*m_F)
        f_k = k_candidates**2 / (np.exp(np.clip((E_k - E_local[remaining])/(kB*T), -500, 500)) + 1)
        u = np.random.uniform(0, k_max[remaining]**2)
        newly_accepted = u < f_k
        idx = np.where(remaining)[0][newly_accepted]
        k_accepted[idx] = k_candidates[newly_accepted]
        accepted[idx] = True

    theta = np.random.uniform(0, np.pi, n)
    phi = np.random.uniform(0, 2*np.pi, n)

    momenta = hbar * k_accepted[:,None] * np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ]).T

    return momenta

def generateCollisionData(
    N_min=30000,
    N_max=200000,
    N_points=10,
    Nratio_min=1,
    Nratio_max=5,
    T_points=2,
    T_max=0.5,
    outfile="collision_data.csv",
):
    """
    Sweeps N in [N_min, N_max] (N_points values) and Nratio in
    [Nratio_min, Nratio_max] (integer steps), and for each combination
    evaluates the collision number/error vs T for tight, weak, and
    ultraweak traps. Saves everything to `outfile` (CSV) instead of plotting.

    Columns: N, Nratio, trap, T_over_Tc, collisions, collision_error
    """
    N_values = np.linspace(N_min, N_max, N_points).astype(int)
    Nratio_values = np.arange(Nratio_min, Nratio_max + 1)
    T = np.linspace(0, T_max, T_points)

    rows = []

    for N in N_values:
        for Nratio in Nratio_values:
            N_f = (int) (N // (Nratio + 1))
            N_b = (int) (N - N_f)

            for trap_name, trap in TRAPS.items():
                mu_f, mu_b = evaluateEnergies(trap[0], trap[1], trap[2], 0, N_f, N_b)
                T_c = evaluateCriticalTemperature(trap[0], trap[1], trap[2], N_b)

                for t in T:
                    _, col, err = evaluateCollisions(
                        mu_f, mu_b, trap[0], trap[1], trap[2], t * T_c, N_f
                    )
                    rows.append(
                        {
                            "N": N,
                            "Nratio": Nratio,
                            "trap": trap_name,
                            "T_over_Tc": t,
                            "collisions": col,
                            "collision_error": err,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")
    return df

def twobodyscatter(k1, k2, n):
    '''
    This function takes in two incoming momenta and returns two outgoing momenta
    according to conservation of energy and momentum, assuming k1 belongs to 3He and
    k2 belongs to 4He. n is the normal vector from 3 to 4
    '''
    kf1 = k1 - 2*m_B/(m_F+m_B)*np.dot(k1-m_F/m_B*k2, n)*n
    kf2 = k1+k2-kf1
    return kf1, kf2

def sampleMomenta(n_k, kx, ky, kz, dkx, dky, dkz, num_samples):
    # cell probabilities (n_k is a density, so multiply by cell volume)
    probs = n_k * dkx * dky * dkz
    probs = probs.ravel()

    # guard against small negative/roundoff values and renormalize
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()

    # sample flat indices according to probs
    flat_indices = np.random.choice(probs.size, size=num_samples, p=probs)

    # convert back to 3D indices
    ix, iy, iz = np.unravel_index(flat_indices, n_k.shape)

    kx_samples = kx[ix, iy, iz]
    ky_samples = ky[ix, iy, iz]
    kz_samples = kz[ix, iy, iz]

    # optional: jitter within each cell (uniform) to smooth out grid discretization
    kx_samples = kx_samples + (np.random.rand(num_samples) - 0.5) * dkx
    ky_samples = ky_samples + (np.random.rand(num_samples) - 0.5) * dky
    kz_samples = kz_samples + (np.random.rand(num_samples) - 0.5) * dkz

    # stack into an (num_samples, 3) array of 3-vectors
    k_vectors = np.stack([kx_samples, ky_samples, kz_samples], axis=-1)

    return k_vectors

def findKspaceDistribution(n, x_vals, y_vals, z_vals):
    psi = np.sqrt(n)
    num = len(x_vals)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    dz = z_vals[1] - z_vals[0]

    psi_k = np.fft.fftn(psi) * (dx * dy * dz)
    psi_k = np.fft.fftshift(psi_k)
    n_k = np.abs(psi_k)**2

    kx = 2*np.pi*np.fft.fftfreq(num, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(num, d=dy)
    kz = 2*np.pi*np.fft.fftfreq(num, d=dz)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    dkx = kx[1,0,0] - kx[0,0,0]
    dky = ky[0,1,0] - ky[0,0,0]
    dkz = kz[0,0,1] - kz[0,0,0]
    n_k *= 1 / (np.sum(n_k) * dkx * dky * dkz)
    return n_k, kx, ky, kz, dkx, dky, dkz

def calculateTemperatureDifferences(N_f, N_b):
    T = np.linspace(0, 0.7, 10)
    mu_b_tight = np.zeros_like(T)
    mu_f_tight = np.zeros_like(T)
    mu_b_weak = np.zeros_like(T)
    mu_f_weak = np.zeros_like(T)
    mu_b_uweak = np.zeros_like(T)
    mu_f_uweak = np.zeros_like(T)
    T_c_tight = evaluateCriticalTemperature(tight[0], tight[1], tight[2], N_b)
    T_c_weak = evaluateCriticalTemperature(weak[0], weak[1], weak[2], N_b)
    T_c_uweak = evaluateCriticalTemperature(ultraweak[0], ultraweak[1], ultraweak[2], N_b)
    ip_b_tight = np.ones_like(T)
    ip_b_weak = np.ones_like(T)
    ip_b_uweak = np.ones_like(T)
    ip_f_tight = np.ones_like(T)
    ip_f_weak = np.ones_like(T)
    ip_f_uweak = np.ones_like(T)
    l2_b_tight = np.zeros_like(T)
    l2_b_weak = np.zeros_like(T)
    l2_b_uweak = np.zeros_like(T)
    l2_f_tight = np.zeros_like(T)
    l2_f_weak = np.zeros_like(T)
    l2_f_uweak = np.zeros_like(T)
    br_tight = np.zeros_like(T)
    br_weak = np.zeros_like(T)
    br_uweak = np.zeros_like(T)
    mu_f_tight[0], mu_b_tight[0] = evaluateEnergies(tight[0], tight[1], tight[2], T[0]*T_c_tight, N_f, N_b)
    mu_f_weak[0], mu_b_weak[0] = evaluateEnergies(weak[0], weak[1], weak[2], T[0]*T_c_weak, N_f, N_b) 
    mu_f_uweak[0], mu_b_uweak[0] = evaluateEnergies(ultraweak[0], ultraweak[1], ultraweak[2], T[0]*T_c_uweak, N_f, N_b)
    x_vals_tight, y_vals_tight, z_vals_tight = generateFullArrays(mu_b_tight[0], mu_f_tight[0], tight[0], tight[1], tight[2], 100, 100)
    x_tight, y_tight, z_tight = np.meshgrid(x_vals_tight, y_vals_tight, z_vals_tight, indexing='ij')
    n_f_tight0, n_b_tight0, n_nc_tight0 = sampleDensity(tight[0], tight[1], tight[2], 0, mu_f_tight[0], mu_b_tight[0], x_tight, y_tight, z_tight)
    x_vals_weak, y_vals_weak, z_vals_weak = generateFullArrays(mu_b_weak[0], mu_f_weak[0], weak[0], weak[1], weak[2], 100, 100)
    x_weak, y_weak, z_weak = np.meshgrid(x_vals_weak, y_vals_weak, z_vals_weak, indexing='ij')
    n_f_weak0, n_b_weak0, n_nc_weak0 = sampleDensity(weak[0], weak[1], weak[2], 0, mu_f_weak[0], mu_b_weak[0], x_weak, y_weak, z_weak)
    x_vals_uweak, y_vals_uweak, z_vals_uweak = generateFullArrays(mu_b_uweak[0], mu_f_uweak[0], ultraweak[0], ultraweak[1], ultraweak[2], 100, 100)
    x_uweak, y_uweak, z_uweak = np.meshgrid(x_vals_uweak, y_vals_uweak, z_vals_uweak, indexing='ij')
    n_f_uweak0, n_b_uweak0, n_nc_uweak0 = sampleDensity(ultraweak[0], ultraweak[1], ultraweak[2], 0, mu_f_uweak[0], mu_b_uweak[0], x_uweak, y_uweak, z_uweak)
    norm_b_tight = evaluateL2Norm(n_b_tight0, np.zeros_like(n_b_tight0), x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])
    norm_b_weak = evaluateL2Norm(n_b_weak0, np.zeros_like(n_b_weak0), x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])
    norm_b_uweak = evaluateL2Norm(n_b_uweak0, np.zeros_like(n_b_uweak0), x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])
    norm_f_tight = evaluateL2Norm(n_f_tight0, np.zeros_like(n_f_tight0), x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])
    norm_f_weak = evaluateL2Norm(n_f_weak0, np.zeros_like(n_f_weak0), x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])
    norm_f_uweak = evaluateL2Norm(n_f_uweak0, np.zeros_like(n_f_uweak0), x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])
    for i in range(1, len(T)):
        mu_f_tight[i], mu_b_tight[i] = evaluateEnergies(tight[0], tight[1], tight[2], T[i]*T_c_tight, N_f, N_b)
        mu_f_weak[i], mu_b_weak[i] = evaluateEnergies(weak[0], weak[1], weak[2], T[i]*T_c_weak, N_f, N_b) 
        mu_f_uweak[i], mu_b_uweak[i] = evaluateEnergies(ultraweak[0], ultraweak[1], ultraweak[2], T[i]*T_c_uweak, N_f, N_b)
        n_f_tight, n_b_tight, n_nc_tight = sampleDensity(tight[0], tight[1], tight[2], T[i]*T_c_tight, mu_f_tight[i], mu_b_tight[i], x_tight, y_tight, z_tight)
        n_f_weak, n_b_weak, n_nc_weak = sampleDensity(weak[0], weak[1], weak[2], T[i]*T_c_weak, mu_f_weak[i], mu_b_weak[i], x_weak, y_weak, z_weak)
        n_f_uweak, n_b_uweak, n_nc_uweak = sampleDensity(ultraweak[0], ultraweak[1], ultraweak[2], T[i]*T_c_uweak, mu_f_uweak[i], mu_b_uweak[i], x_uweak, y_uweak, z_uweak)
        ip_b_tight[i] = evaluateNormalisedInnerProduct(n_b_tight, n_b_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])
        ip_f_tight[i] = evaluateNormalisedInnerProduct(n_f_tight, n_f_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])
        ip_b_weak[i] = evaluateNormalisedInnerProduct(n_b_weak, n_b_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])
        ip_f_weak[i] = evaluateNormalisedInnerProduct(n_f_weak, n_f_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])
        ip_b_uweak[i] = evaluateNormalisedInnerProduct(n_b_uweak, n_b_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])
        ip_f_uweak[i] = evaluateNormalisedInnerProduct(n_f_uweak, n_f_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])
        l2_b_tight[i] = evaluateL2Norm(n_b_tight, n_b_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])/norm_b_tight
        l2_f_tight[i] = evaluateL2Norm(n_f_tight, n_f_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])/norm_f_tight
        l2_b_weak[i] = evaluateL2Norm(n_b_weak, n_b_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])/norm_b_weak
        l2_f_weak[i] = evaluateL2Norm(n_f_weak, n_f_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])/norm_f_weak
        l2_b_uweak[i] = evaluateL2Norm(n_b_uweak, n_b_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])/norm_b_uweak
        l2_f_uweak[i] = evaluateL2Norm(n_f_uweak, n_f_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])/norm_f_uweak
        br_tight[i] = integrateDensity(n_nc_tight, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])/N_b
        br_weak[i] = integrateDensity(n_nc_weak, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])/N_b
        br_uweak[i] = integrateDensity(n_nc_uweak, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])/N_b
    results = {
        "T": T.tolist(),
        "inner_product": {
            "boson": {
                "tight": ip_b_tight.tolist(),
                "weak": ip_b_weak.tolist(),
                "uweak": ip_b_uweak.tolist(),
            },
            "fermion": {
                "tight": ip_f_tight.tolist(),
                "weak": ip_f_weak.tolist(),
                "uweak": ip_f_uweak.tolist(),
            },
        },
        "l2_norm": {
            "boson": {
                "tight": l2_b_tight.tolist(),
                "weak": l2_b_weak.tolist(),
                "uweak": l2_b_uweak.tolist(),
            },
            "fermion": {
                "tight": l2_f_tight.tolist(),
                "weak": l2_f_weak.tolist(),
                "uweak": l2_f_uweak.tolist(),
            },
        },
        "boson_number_ratio": {
            "tight": br_tight.tolist(),
            "weak": br_weak.tolist(),
            "uweak": br_uweak.tolist(),
        },
    }

    with open("temperature_differences.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

rng = np.random.default_rng(seed=1)

def drawNumberFractionTransfer(N, Nerr, F, Ferr, T, Terr):
    return rng.normal(loc = N, scale=Nerr), rng.normal(loc = F, scale=Ferr), rng.normal(loc = T, scale=Terr)

def binned_medians(filename, bin_width=1000, f_lo=0, f_hi=1, N_min=2000, N_max=None,
                   min_shots=10, n_boot=1000, y="C"):
    """Returns N_med, N_err, y_med, y_err, counts (one entry per fixed-width N bin)."""
    d = np.loadtxt(filename, delimiter=",", skiprows=1)
    d = d[~np.isnan(d[:, :3]).any(axis=1)]
    d = d[(d[:, 1] > 0) & (d[:, 2] > f_lo) & (d[:, 2] < f_hi)]

    C = d[:, 0]*2            # true collisions
    N = d[:, 1]*4            # true atom number
    val = C/N**2 if y == "C/N2" else C

    if N_max is None:
        N_max = N.max()
    edges = np.arange(N_min, N_max + bin_width, bin_width)
    idx = np.digitize(N, edges) - 1

    rng = np.random.default_rng(0)
    def med_err(x):
        boots = [np.median(rng.choice(x, len(x))) for _ in range(n_boot)]
        return np.median(x), np.std(boots)

    out = []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < min_shots:
            continue
        Nm, Ne = med_err(N[m])
        vm, ve = med_err(val[m])
        out.append((Nm, Ne, vm, ve, m.sum()))
    return np.array(out).T if out else (np.array([]),)*5
