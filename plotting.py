from core import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'axes.formatter.use_mathtext': True,   # fixes the minus-sign / tick-label glyph warnings with cmr10
    'axes.unicode_minus': False,           # cmr10 lacks the unicode minus glyph
})
plt.rc('font', size=12)          # Default text size
plt.rc('axes', titlesize=25)     # Axes title size
plt.rc('axes', labelsize=25)     # X and Y label size
plt.rc('xtick', labelsize=15)    # X-tick label size
plt.rc('ytick', labelsize=15)    # Y-tick label size
plt.rc('legend', fontsize=20)    # Legend size


def plotBosonMomentumProfile(N, Nratio, mu_b, mu_f, fx, fy, fz, cmapChoice="grayPurple"):

    N_f = (int) (N//(Nratio+1))
    N_b = (int) (N-N_f)
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    k_f = np.sqrt(m_B/m_F)*wbar*(6*N_f)**(1/3)
    sampling_rate = 4*k_f
    distance = 1/sampling_rate
    size = 100/k_f
    num = (int) (size//distance)
    x_vals = np.linspace(-size/2, size/2, num)
    y_vals = x_vals
    z_vals = x_vals
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    n_f, n_b, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, x, y, z)

    n_k, kx, ky, kz, dkx, dky, dkz = findKspaceDistribution(n_b, x_vals, y_vals, z_vals)

    # colormaps: pick one
    cmaps = {
        "grayPurple": LinearSegmentedColormap.from_list("grayPurple", [lightGray, dullPurple]),
        "purpleGold": LinearSegmentedColormap.from_list("purpleGold", [dullPurple, gold]),
    }
    cmap = cmaps[cmapChoice]

    lenX, lenY = n_k.shape[0], n_k.shape[1]
    sliceYZ = n_k[lenX//2, :, :]
    sliceXZ = n_k[:, lenY//2, :]

    # shared scale so a single colorbar is valid for both panels
    vmax = max(sliceYZ.max(), sliceXZ.max())

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    im1 = ax[0].imshow(sliceYZ.T, origin="lower",
                       extent=[ky.min(), ky.max(), kz.min(), kz.max()],
                       aspect="equal", cmap=cmap, vmin=0, vmax=vmax)
    ax[0].set_title("Boson Momentum Y-Z Plane")
    ax[0].set_xlabel(r"$k_y\ (\mu m^{-1})$")
    ax[0].set_ylabel(r"$k_z\ (\mu m^{-1})$")

    ax[1].imshow(sliceXZ.T, origin="lower",
                 extent=[kx.min(), kx.max(), kz.min(), kz.max()],
                 aspect="equal", cmap=cmap, vmin=0, vmax=vmax)
    ax[1].set_title("Boson Momentum X-Z Plane")
    ax[1].set_xlabel(r"$k_x\ (\mu m^{-1})$")
    ax[1].set_ylabel(r"$k_z\ (\mu m^{-1})$")

    # one colorbar spanning both panels
    fig.colorbar(im1, ax=ax, shrink=0.8, label="Momentum density (arb.)")

    for a in ax:
        a.add_patch(Circle((0, 0), k_f, fill=False, edgecolor='black',
                           linestyle='--', linewidth=1.5))

    fig.savefig('bosonmomentum.pdf', bbox_inches='tight')
    plt.show()

fx, fy, fz = tight[0], tight[1], tight[2]
T = 0
Nratio = 3
N = 40000
N_f = N/(1+Nratio)
N_b = N-N_f
mu_f, mu_b =evaluateEnergies(fx, fy, fz, T, N_f, N_b)
plotBosonMomentumProfile(N, Nratio,mu_b, mu_f, fx, fy, fz)

def plotDensityProfiles(mu_b, mu_f, fx, fy, fz, T, numX, numY):

    X, Y, Z = generateFullArrays(mu_b, mu_f, fx, fy, fz, numX, numY)

    midX = len(X) // 2
    midY = len(Y) // 2

    # YZ slice: fix x at midpoint
    yy, zz = np.meshgrid(Y, Z, indexing='ij')
    x_plane = np.full_like(yy, X[midX])
    n_F_yz, n_B_yz, n_NC_yz = sampleDensity(fx, fy, fz, T, mu_f, mu_b, x_plane, yy, zz)

    # XZ slice: fix y at midpoint
    xx, zz2 = np.meshgrid(X, Z, indexing='ij')
    y_plane = np.full_like(xx, Y[midY])
    n_F_xz, n_B_xz, n_NC_xz = sampleDensity(fx, fy, fz, T, mu_f, mu_b, xx, y_plane, zz2)

    # colormaps: medGray at zero density, up to the species colour
    cmapB = LinearSegmentedColormap.from_list("grayGold", [lightGray, gold])
    cmapF = LinearSegmentedColormap.from_list("grayPurple", [lightGray, dullPurple])

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.patch.set_facecolor((1, 1, 1))

    # share a colour scale between the two slices of each species,
    # starting at exactly 0 so medGray always means zero density
    vmaxF = max(n_F_yz.max(), n_F_xz.max())
    vmaxB = max(n_B_yz.max(), n_B_xz.max())

    im1 = ax[0][0].imshow(n_F_yz.T, origin="lower",
                          extent=[Y[0], Y[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapF, vmin=0, vmax=vmaxF)
    ax[0][0].set_title("Fermion Density Y-Z Plane")
    ax[0][0].set_xlabel(r"$y\ (\mu m)$")
    ax[0][0].set_ylabel(r"$z\ (\mu m)$")

    im2 = ax[0][1].imshow(n_F_xz.T, origin="lower",
                          extent=[X[0], X[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapF, vmin=0, vmax=vmaxF)
    ax[0][1].set_title("Fermion Density X-Z Plane")
    ax[0][1].set_xlabel(r"$x\ (\mu m)$")
    ax[0][1].set_ylabel(r"$z\ (\mu m)$")

    im3 = ax[1][0].imshow(n_B_yz.T, origin="lower",
                          extent=[Y[0], Y[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxB)
    ax[1][0].set_title("Boson Density Y-Z Plane")
    ax[1][0].set_xlabel(r"$y\ (\mu m)$")
    ax[1][0].set_ylabel(r"$z\ (\mu m)$")

    im4 = ax[1][1].imshow(n_B_xz.T, origin="lower",
                          extent=[X[0], X[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxB)
    ax[1][1].set_title("Boson Density X-Z Plane")
    ax[1][1].set_xlabel(r"$x\ (\mu m)$")
    ax[1][1].set_ylabel(r"$z\ (\mu m)$")



    fig.colorbar(im1, ax=ax[0, :], label=r"$n_F\ (\mu m^{-3})$")
    fig.colorbar(im3, ax=ax[1, :], label=r"$n_B\ (\mu m^{-3})$")

    plt.savefig('densityprofiles.pdf', bbox_inches='tight')
    plt.show()




def plotDensityProfilesWithThermal(mu_b, mu_f, fx, fy, fz, T, numX, numY):

    X, Y, Z = generateFullArrays(mu_b, mu_f, fx, fy, fz, numX, numY)

    midX = len(X) // 2
    midY = len(Y) // 2

    # YZ slice: fix x at midpoint
    yy, zz = np.meshgrid(Y, Z, indexing='ij')
    x_plane = np.full_like(yy, X[midX])
    n_F_yz, n_B_yz, n_NC_yz = sampleDensity(fx, fy, fz, T, mu_f, mu_b, x_plane, yy, zz)

    # XZ slice: fix y at midpoint
    xx, zz2 = np.meshgrid(X, Z, indexing='ij')
    y_plane = np.full_like(xx, Y[midY])
    n_F_xz, n_B_xz, n_NC_xz = sampleDensity(fx, fy, fz, T, mu_f, mu_b, xx, y_plane, zz2)

    # colormaps: medGray at zero density, up to the species colour
    cmapB = LinearSegmentedColormap.from_list("grayGold", [lightGray, gold])
    cmapF = LinearSegmentedColormap.from_list("grayPurple", [lightGray, dullPurple])

    # shared colour scale per row, starting at exactly 0
    vmaxF = max(n_F_yz.max(), n_F_xz.max())
    vmaxB = max(n_B_yz.max(), n_B_xz.max())
    vmaxNC = max(n_NC_yz.max(), n_NC_xz.max())

    fig, ax = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    fig.patch.set_facecolor((1, 1, 1))

    im1 = ax[0][0].imshow(n_F_yz.T, origin="lower",
                          extent=[Y[0], Y[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapF, vmin=0, vmax=vmaxF)
    ax[0][0].set_title("Fermion Density Y-Z Plane")
    ax[0][0].set_xlabel(r"$y\ (\mu m)$")
    ax[0][0].set_ylabel(r"$z\ (\mu m)$")

    im2 = ax[0][1].imshow(n_F_xz.T, origin="lower",
                          extent=[X[0], X[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapF, vmin=0, vmax=vmaxF)
    ax[0][1].set_title("Fermion Density X-Z Plane")
    ax[0][1].set_xlabel(r"$x\ (\mu m)$")
    ax[0][1].set_ylabel(r"$z\ (\mu m)$")

    im3 = ax[1][0].imshow(n_B_yz.T, origin="lower",
                          extent=[Y[0], Y[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxB)
    ax[1][0].set_title("Boson Density Y-Z Plane")
    ax[1][0].set_xlabel(r"$y\ (\mu m)$")
    ax[1][0].set_ylabel(r"$z\ (\mu m)$")

    im4 = ax[1][1].imshow(n_B_xz.T, origin="lower",
                          extent=[X[0], X[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxB)
    ax[1][1].set_title("Boson Density X-Z Plane")
    ax[1][1].set_xlabel(r"$x\ (\mu m)$")
    ax[1][1].set_ylabel(r"$z\ (\mu m)$")

    im5 = ax[2][0].imshow(n_NC_yz.T, origin="lower",
                          extent=[Y[0], Y[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxNC)
    ax[2][0].set_title("Thermal Boson Density Y-Z Plane")
    ax[2][0].set_xlabel(r"$y\ (\mu m)$")
    ax[2][0].set_ylabel(r"$z\ (\mu m)$")

    im6 = ax[2][1].imshow(n_NC_xz.T, origin="lower",
                          extent=[X[0], X[-1], Z[0], Z[-1]],
                          aspect="equal", cmap=cmapB, vmin=0, vmax=vmaxNC)
    ax[2][1].set_title("Thermal Boson Density X-Z Plane")
    ax[2][1].set_xlabel(r"$x\ (\mu m)$")
    ax[2][1].set_ylabel(r"$z\ (\mu m)$")

    # one colourbar per row
    fig.colorbar(im1, ax=ax[0, :], label=r"$n_F\ (\mu m^{-3})$")
    fig.colorbar(im3, ax=ax[1, :], label=r"$n_B\ (\mu m^{-3})$")
    fig.colorbar(im5, ax=ax[2, :], label=r"$n_{th}\ (\mu m^{-3})$")

    plt.savefig('densityprofileswiththermal.pdf', bbox_inches='tight')
    plt.show()

# fx, fy, fz = tight[0], tight[1], tight[2]
# T_c = evaluateCriticalTemperature(fx, fy, fz, 30000)
# T_f = evaluateFermiTemperature(fx, fy, fz, 10000)
# print(0.6*T_c/T_f)
# # mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0.6*T_c, 10000, 30000)
# # plotDensityProfilesWithThermal(mu_b, mu_f, fx, fy, fz, 0.6*T_c, 200, 200)

def plotCollisions(x="N", N=None, Nratio=None, T_over_Tc=None,
                   filename="collision_data.csv", traps=None, save=True):
    '''
    Plots the simulated collision number for each trap against one variable,
    holding the other two fixed.

    x         : "N", "Nratio" or "T_over_Tc", the variable on the x-axis
    N, Nratio, T_over_Tc : values to hold fixed for the other two variables
                           (the one chosen as x is ignored)
    traps     : list of trap names to plot, default all in the file
    '''
    df = pd.read_csv(filename)

    fixed = {"N": N, "Nratio": Nratio, "T_over_Tc": T_over_Tc}
    if x not in fixed:
        raise ValueError(f"x must be one of {list(fixed)}")

    # apply the fixed values, with a tolerance for floating point
    for name, val in fixed.items():
        if name == x:
            continue
        if val is None:
            raise ValueError(f"Give a value for {name}. Available: {sorted(df[name].unique())}")
        df = df[np.isclose(df[name], val)]
    if df.empty:
        raise ValueError("No rows match those fixed values.")

    colours = {"tight": dullPurple, "weak": medGray, "ultraweak": gold}
    names = {"tight": "Tight", "weak": "Weak", "ultraweak": "Ultraweak"}
    labels = {"N": "Total Atom Number", "Nratio": r"$N_b/N_f$", "T_over_Tc": r"$T/T_c^*$"}

    if traps is None:
        traps = [t for t in colours if t in df["trap"].unique()]

    fig, ax = plt.subplots(figsize=(8,5))
    for trap in traps:
        d = df[df["trap"] == trap].sort_values(x)
        ax.errorbar(d[x], d["collisions"], yerr=d["collision_error"],
                    color=colours[trap], marker="o", markersize=4,
                    linestyle="none", elinewidth=1.5, capsize=2, capthick=1.5,
                    label=names[trap])

    ax.set_xlabel(labels[x])
    ax.set_ylabel("Number of Collisions")
    ax.legend()
    fig.tight_layout()
    if save:
        tag = "_".join(f"{k}{v}" for k, v in fixed.items() if k != x and v is not None)
        fig.savefig(f"collisions_vs_{x}_{tag}.pdf", bbox_inches="tight")
    plt.show()

def loadCollisionData(infile="collision_data.csv"):
    """Loads a previously generated collision data CSV into a DataFrame."""
    return pd.read_csv(infile)


def _nearest_value(series, target):
    """Returns the value in `series` closest to `target` (handles float T mismatches)."""
    values = series.unique()
    return values[np.argmin(np.abs(values - target))]


def plotCollisionsVsN(df, Nratio, T_over_Tc, traps=None, outfile=None):
    """
    2D plot of collisions vs N, at fixed Nratio and fixed T/Tc*.
    `traps` is an optional list of trap names to include (default: all three).
    """
    traps = traps or list(TRAPS.keys())
    t_val = _nearest_value(df["T_over_Tc"], T_over_Tc)

    subset = df[(df["Nratio"] == Nratio) & (df["T_over_Tc"] == t_val)]

    plt.figure()
    for trap_name in traps:
        d = subset[subset["trap"] == trap_name].sort_values("N")
        if d.empty:
            continue
        plt.errorbar(
            d["N"],
            d["collisions"],
            d["collision_error"],
            label=trap_name.capitalize(),
            color=TRAP_COLORS.get(trap_name),
            ecolor=TRAP_COLORS.get(trap_name),
            fmt='o'
        )

    plt.xlabel("Total Atom Number")
    plt.ylabel("Number of Collisions")
    plt.legend()
    if outfile:
        plt.savefig(outfile+".pdf", bbox_inches='tight')
    plt.show()

def plotCollisionsPerFermionVsN(df, Nratio, T_over_Tc, traps=None, outfile=None):
    """
    2D plot of collisions vs N, at fixed Nratio and fixed T/Tc*.
    `traps` is an optional list of trap names to include (default: all three).
    """
    
    traps = traps or list(TRAPS.keys())
    t_val = _nearest_value(df["T_over_Tc"], T_over_Tc)

    subset = df[(df["Nratio"] == Nratio) & (df["T_over_Tc"] == t_val)]

    plt.figure()
    for trap_name in traps:
        
        d = subset[subset["trap"] == trap_name].sort_values("N")

        if d.empty:
            continue
        N = d["N"]
        N_f = N/(Nratio+1)
        plt.errorbar(
            d["N"],
            d["collisions"]/N_f,
            d["collision_error"]/N_f,
            label=trap_name.capitalize(),
            color=TRAP_COLORS.get(trap_name),
            ecolor=TRAP_COLORS.get(trap_name),
            fmt='o'
        )

    plt.xlabel("N")
    plt.ylabel("Collisions Per Fermion")
    plt.legend()
    if outfile:
        plt.savefig(outfile+".pdf", bbox_inches='tight')
    plt.show()


def plotCollisionsVsNratio(df, N, T_over_Tc, traps=None, outfile=None):
    """
    2D plot of collisions vs Nratio, at fixed N and fixed T/Tc*.
    `traps` is an optional list of trap names to include (default: all three).
    """
    traps = traps or list(TRAPS.keys())
    N_val = _nearest_value(df["N"], N)
    t_val = _nearest_value(df["T_over_Tc"], T_over_Tc)

    subset = df[(df["N"] == N_val) & (df["T_over_Tc"] == t_val)]

    plt.figure()
    for trap_name in traps:
        d = subset[subset["trap"] == trap_name].sort_values("Nratio")
        if d.empty:
            continue
        plt.errorbar(
            d["Nratio"],
            d["collisions"],
            d["collision_error"],
            label=trap_name.capitalize(),
            color=TRAP_COLORS.get(trap_name),
            ecolor=TRAP_COLORS.get(trap_name),
            fmt='o',
        )

    plt.xlabel(f"$N_b/N_f$")
    plt.ylabel("Number of Collisions")
    plt.legend()
    if outfile:
        plt.savefig(outfile+".pdf", bbox_inches='tight')
    plt.show()

def plotCollisionsPerFermionVsNratio(df, N, T_over_Tc, traps=None, outfile=None):
    """
    2D plot of collisions vs Nratio, at fixed N and fixed T/Tc*.
    `traps` is an optional list of trap names to include (default: all three).
    """
    traps = traps or list(TRAPS.keys())
    N_val = _nearest_value(df["N"], N)
    t_val = _nearest_value(df["T_over_Tc"], T_over_Tc)

    subset = df[(df["N"] == N_val) & (df["T_over_Tc"] == t_val)]

    plt.figure()
    for trap_name in traps:
        d = subset[subset["trap"] == trap_name].sort_values("Nratio")
        if d.empty:
            continue
        Nratio = d["Nratio"]
        N_f = N/(Nratio+1)
        plt.errorbar(
            d["Nratio"],
            d["collisions"]/N_f,
            d["collision_error"]/N_f,
            label=trap_name.capitalize(),
            color=TRAP_COLORS.get(trap_name),
            ecolor=TRAP_COLORS.get(trap_name),
        )

    plt.xlabel(f"$N_b/N_f$")
    plt.ylabel("Collisions Per Fermion")
    plt.legend()
    if outfile:
        plt.savefig(outfile+".pdf", bbox_inches='tight')
    plt.show()


def plotCollisionsVsT(df, N, Nratio, traps=None, outfile=None):
    """
    Bonus: reproduces the behavior of your original `run` function
    (collisions vs T/Tc*), but reading from the saved data instead of
    recomputing, at fixed N and Nratio.
    """
    traps = traps or list(TRAPS.keys())
    N_val = _nearest_value(df["N"], N)

    subset = df[(df["N"] == N_val) & (df["Nratio"] == Nratio)]

    plt.figure()
    for trap_name in traps:
        d = subset[subset["trap"] == trap_name].sort_values("T_over_Tc")
        if d.empty:
            continue
        plt.errorbar(
            d["T_over_Tc"],
            d["collisions"],
            d["collision_error"],
            label=trap_name.capitalize(),
            color=TRAP_COLORS.get(trap_name),
            ecolor=TRAP_COLORS.get(trap_name),
        )

    plt.xlabel(r"$T/T_C^*$")
    plt.ylabel("Number of Collisions")
    plt.legend()
    if outfile:
        plt.savefig(outfile+".pdf", bbox_inches='tight')
    plt.show()

from matplotlib.patches import Circle
from matplotlib.widgets import Slider

def evaluateDetectorImageMultiFreq(T, N, Nratio):
    """
    Same physics/simulation as evaluateDetectorImage, but run once for each
    of three (fx, fy, fz) trapping-frequency sets, and plotted as three
    side-by-side XY panels (one per frequency set) instead of XY/XT/YT.

    Parameters
    ----------
    T, N, Nratio : same as before, shared across all three runs

    Returns
    -------
    results : list of dicts, one per frequency set, each containing
        XYT_detected, species_detected, fermionDetection, bosonDetection,
        collisions, and the (fx, fy, fz) used.
    """

    def runOne(fx, fy, fz):
        N_f = (int)(N // (Nratio + 1))
        N_b = (int)(N - N_f)
        mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, N_f, N_b)
        T_c = evaluateCriticalTemperature(fx, fy, fz, N_b)
        wbar = 2 * np.pi * (fx * fy * fz) ** (1 / 3)
        k_f = np.sqrt(m_B / m_F) * wbar * (6 * N_f) ** (1 / 3)
        sampling_rate = 4 * k_f
        distance = 1 / sampling_rate
        size = 100 / k_f
        num = (int)(size // distance)
        x_vals = np.linspace(-size / 2, size / 2, num)
        y_vals = x_vals
        z_vals = x_vals
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        n_f, n_b, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, x, y, z)

        n_k, kx, ky, kz, dkx, dky, dkz = findKspaceDistribution(n_b, x_vals, y_vals, z_vals)

        _, _, collisions, _ = evaluateCollisions(mu_f, mu_b, fx, fy, fz, T * T_c, N_f)
        collisions = int(collisions)
        boson_k_samples = sampleMomenta(n_k, kx, ky, kz, dkx, dky, dkz, collisions)
        fp = sampleFermionPositions(x_vals, y_vals, z_vals, n_f, N_f, collisions)
        fermion_k_samples = sampleFermionMomenta(mu_b, mu_f, T * T_c, fx, fy, fz, fp) / hbar

        XYT = np.zeros((2 * collisions, 3))
        species = np.zeros(2 * collisions, dtype=int)  # 0 = fermion (3He), 1 = boson (4He)

        for i in range(collisions):
            bvi = hbar*boson_k_samples[i]/m_B
            bvi[2] -= np.sqrt(2) * vrec
            bki = bvi*m_B/hbar
            fki = fermion_k_samples[i]
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            phi = np.random.uniform(0, 2 * np.pi)
            n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), costheta])
            fkf, bkf = twobodyscatter(fki, bki, n)
            fv = fkf / m_F * hbar
            bv = bkf / m_B * hbar
            s = -848000  # distance to detector plate
            tf = (fv[2] + np.sqrt(fv[2] ** 2 - 2 * 9.81 * 1e-6 * s)) / (9.81 * 1e-6)
            tb = (bv[2] + np.sqrt(bv[2] ** 2 - 2 * 9.81 * 1e-6 * s)) / (9.81 * 1e-6)
            xf = fv[0] * tf
            yf = fv[1] * tf
            xb = bv[0] * tb
            yb = bv[1] * tb
            XYT[2 * i] = np.array([xf, yf, tf])
            XYT[2 * i + 1] = np.array([xb, yb, tb])
            species[2 * i] = 0
            species[2 * i + 1] = 1

        # eliminate any which fall outside the detector bounds
        distances = np.sqrt(XYT[:, 0] ** 2 + XYT[:, 1] ** 2)
        detected = distances < 40000
        XYT_detected = XYT[detected]
        species_detected = species[detected]

        is_f = species_detected == 0
        is_b = species_detected == 1
        fermionsDetected = XYT_detected[is_f].shape[0]
        bosonsDetected = XYT_detected[is_b].shape[0]
        fermionDetection = fermionsDetected / collisions
        bosonDetection = bosonsDetected / collisions

        # randomly get rid of 75% due to detector inefficiency
        keep_mask = np.random.uniform(0, 1, size=XYT_detected.shape[0]) > 0.75
        XYT_detected = XYT_detected[keep_mask]
        species_detected = species_detected[keep_mask]

        return {
            'XYT_detected': XYT_detected,
            'species_detected': species_detected,
            'fermionDetection': fermionDetection,
            'bosonDetection': bosonDetection,
            'collisions': collisions,
            'fx': fx, 'fy': fy, 'fz': fz,
        }

    results = [runOne(tight[0], tight[1], tight[2]), runOne(weak[0], weak[1], weak[2]), runOne(ultraweak[0], ultraweak[1], ultraweak[2])]

    detector_radius = 40000  # 40 mm in µm

    fig = plt.figure(figsize=(18, 6))
    ax_panels = [
        fig.add_axes([0.05, 0.15, 0.28, 0.75]),
        fig.add_axes([0.38, 0.15, 0.28, 0.75]),
        fig.add_axes([0.70, 0.15, 0.28, 0.75]),
    ]

    def draw_panel(ax, res):
        XYT_detected = res['XYT_detected']
        species_detected = res['species_detected']
        is_f = species_detected == 0
        is_b = species_detected == 1
        ax.scatter(XYT_detected[is_f, 0], XYT_detected[is_f, 1], s=2, alpha=1,
                   color=dullPurple, label='³He')
        ax.scatter(XYT_detected[is_b, 0], XYT_detected[is_b, 1], s=2, alpha=1,
                   color=gold, label='⁴He')
        circle = Circle((0, 0), detector_radius, fill=False, edgecolor='black',
                         linestyle='--', linewidth=1.5, label='Detector edge')
        ax.add_patch(circle)
        ax.set_xlabel(rf'$x (\mu m)$')
        ax.set_ylabel(rf'$y (\mu m)$')
        ax.set_aspect('equal')
        ax.set_xlim(-1.2 * detector_radius, 1.2 * detector_radius)
        ax.set_ylim(-1.2 * detector_radius, 1.2 * detector_radius)
        ax.legend(loc='upper right')

    for ax, res in zip(ax_panels, results):
        draw_panel(ax, res)

    plt.show()

    return results


def evaluateDetectorImage(fx, fy, fz, T, N, Nratio):
    #First, we need to evaluate the momentum distribution of the BEC
    #this requires knowing the necessary sampling rate
    N_f = (int) (N//(Nratio+1))
    N_b = (int) (N-N_f)
    mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, N_f, N_b)
    T_c = evaluateCriticalTemperature(fx, fy, fz, N_b)
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    k_f = np.sqrt(m_B/m_F)*wbar*(6*N_f)**(1/3) #we want to sample momenta up to the fermi momenta
    sampling_rate = 4*k_f
    distance = 1/(sampling_rate)
    size = 100/k_f
    num = (int) (size//distance)
    x_vals = np.linspace(-size/2, size/2, num)
    y_vals = x_vals
    z_vals = x_vals
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    n_f, n_b, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, x, y, z)

    n_k, kx, ky, kz, dkx, dky, dkz = findKspaceDistribution(n_b, x_vals, y_vals, z_vals)

    _, _, collisions, _ = evaluateCollisions(mu_f, mu_b, fx, fy, fz, T*T_c, N_f)
    collisions = int(collisions)
    boson_k_samples = sampleMomenta(n_k, kx, ky, kz, dkx, dky, dkz, collisions)
    fp = sampleFermionPositions(x_vals, y_vals, z_vals, n_f, N_f, collisions)
    fermion_k_samples = sampleFermionMomenta(mu_b, mu_f, T*T_c, fx, fy, fz, fp)/hbar

    XYT = np.zeros((2*collisions, 3))
    species = np.zeros(2*collisions, dtype=int)  # 0 = fermion (3He), 1 = boson (4He)

    for i in range(collisions):
        bvi = hbar*boson_k_samples[i]/m_B
        bvi[2] -= np.sqrt(2) * vrec
        bki = bvi*m_B/hbar
        fki = fermion_k_samples[i]
        costheta = np.random.uniform(-1,1)
        theta = np.arccos(costheta)
        phi = np.random.uniform(0, 2*np.pi)
        n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), costheta])
        fkf, bkf = twobodyscatter(fki, bki, n)
        fv = fkf/m_F*hbar
        bv = bkf/m_B*hbar
        s = -848000 #distance to detector plate
        tf = (fv[2] + np.sqrt(fv[2]**2-2*9.81*1e-6*s))/(9.81*1e-6)
        tb = (bv[2] + np.sqrt(bv[2]**2-2*9.81*1e-6*s))/(9.81*1e-6)
        xf = fv[0]*tf
        yf = fv[1]*tf
        xb = bv[0]*tb
        yb = bv[1]*tb
        XYT[2*i] = np.array([xf, yf, tf])
        XYT[2*i+1] = np.array([xb, yb, tb])
        species[2*i] = 0
        species[2*i+1] = 1

    #first, we eliminate any which fall outside the detector bounds
    distances = np.sqrt(XYT[:, 0]**2 + XYT[:, 1]**2)
    detected = distances < 40000
    XYT_detected = XYT[detected]
    species = species[detected]
    



    is_f = species == 0
    is_b = species == 1
    fermionsDetected = XYT_detected[is_f].shape[0]
    bosonsDetected = XYT_detected[is_b].shape[0]
    fermionDetection = fermionsDetected/collisions
    bosonDetection = bosonsDetected/collisions

    #now randomly get rid of 75% due to detector inefficiency
    keep_mask = np.random.uniform(0, 1, size=XYT_detected.shape[0]) > 0.75
    XYT_detected = XYT_detected[keep_mask]
    species_detected = species[keep_mask]
    is_f = species_detected == 0
    is_b = species_detected == 1

    t_all = XYT_detected[:,2]
    t_min, t_max_data = np.nanmin(t_all), np.nanmax(t_all)

    #now plot the XY plane, XT plane and YT plane
    fig = plt.figure(figsize=(15, 6))
    ax_xy = fig.add_axes([0.05, 0.15, 0.28, 0.75])
    ax_xt = fig.add_axes([0.38, 0.15, 0.28, 0.75])
    ax_yt = fig.add_axes([0.70, 0.15, 0.28, 0.75])
    ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.03])

    detector_radius = 40000  # 40 mm in µm

    def draw_xy(t_cutoff):
        ax_xy.clear()
        mask_t = t_all <= t_cutoff
        fmask = is_f & mask_t
        bmask = is_b & mask_t
        ax_xy.scatter(XYT_detected[fmask,0], XYT_detected[fmask,1], s=2, alpha=0.4, color=dullPurple, label='³He')
        ax_xy.scatter(XYT_detected[bmask,0], XYT_detected[bmask,1], s=2, alpha=0.4, color=gold, label='⁴He')
        circle = Circle((0, 0), detector_radius, fill=False, edgecolor='black',
                         linestyle='--', linewidth=1.5, label='Detector edge')
        ax_xy.add_patch(circle)
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_title(f'XY plane (t ≤ {t_cutoff:.1f})')
        ax_xy.set_aspect('equal')
        ax_xy.set_xlim(-1.2*detector_radius, 1.2*detector_radius)
        ax_xy.set_ylim(-1.2*detector_radius, 1.2*detector_radius)
        ax_xy.legend(loc='upper right')

    draw_xy(t_max_data)

    ax_xt.scatter(XYT_detected[is_f,0], XYT_detected[is_f,2], s=2, alpha=0.4, color=dullPurple, label='³He')
    ax_xt.scatter(XYT_detected[is_b,0], XYT_detected[is_b,2], s=2, alpha=0.4, color='tab:orange', label='⁴He')
    ax_xt.set_xlabel('x')
    ax_xt.set_ylabel('t')
    ax_xt.set_title('XT plane')
    ax_xt.legend()

    ax_yt.scatter(XYT_detected[is_f,1], XYT_detected[is_f,2], s=2, alpha=0.4, color='tab:blue', label='³He')
    ax_yt.scatter(XYT_detected[is_b,1], XYT_detected[is_b,2], s=2, alpha=0.4, color='tab:orange', label='⁴He')
    ax_yt.set_xlabel('y')
    ax_yt.set_ylabel('t')
    ax_yt.set_title('YT plane')
    ax_yt.legend()

    slider = Slider(ax_slider, 't cutoff', t_min, t_max_data, valinit=t_max_data)

    def update(val):
        draw_xy(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    return XYT_detected, species_detected, fermionDetection,  bosonDetection, collisions

def evaluateDetectorImageXY(fx, fy, fz, T, N, Nratio):
    # First, we need to evaluate the momentum distribution of the BEC
    # this requires knowing the necessary sampling rate
    N_f = int(N // (Nratio + 1))
    N_b = int(N - N_f)
    mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, N_f, N_b)
    T_c = evaluateCriticalTemperature(fx, fy, fz, N_b)
    wbar = 2*np.pi*(fx*fy*fz)**(1/3)
    k_f = np.sqrt(m_B/m_F)*wbar*(6*N_f)**(1/3)  # sample momenta up to the Fermi momentum
    sampling_rate = 4*k_f
    distance = 1/sampling_rate
    size = 100/k_f
    num = int(size//distance)
    x_vals = np.linspace(-size/2, size/2, num)
    y_vals = x_vals
    z_vals = x_vals
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    n_f, n_b, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, x, y, z)

    n_k, kx, ky, kz, dkx, dky, dkz = findKspaceDistribution(n_b, x_vals, y_vals, z_vals)

    _, _, collisions, _ = evaluateCollisions(mu_f, mu_b, fx, fy, fz, T*T_c, N_f)
    collisions = int(collisions)
    boson_k_samples = sampleMomenta(n_k, kx, ky, kz, dkx, dky, dkz, collisions)
    fp = sampleFermionPositions(x_vals, y_vals, z_vals, n_f, N_f, collisions)
    fermion_k_samples = sampleFermionMomenta(mu_b, mu_f, T*T_c, fx, fy, fz, fp)/hbar

    XYT = np.zeros((2*collisions, 3))
    species = np.zeros(2*collisions, dtype=int)  # 0 = fermion (3He), 1 = boson (4He)

    for i in range(collisions):
        bvi = hbar*boson_k_samples[i]/m_B
        bvi[2] -= np.sqrt(2) * vrec
        bki = bvi*m_B/hbar
        fki = fermion_k_samples[i]
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        phi = np.random.uniform(0, 2*np.pi)
        n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), costheta])
        fkf, bkf = twobodyscatter(fki, bki, n)
        fv = fkf/m_F*hbar
        bv = bkf/m_B*hbar
        s = -848000  # distance to detector plate
        tf = (fv[2] + np.sqrt(fv[2]**2 - 2*9.81*1e-6*s))/(9.81*1e-6)
        tb = (bv[2] + np.sqrt(bv[2]**2 - 2*9.81*1e-6*s))/(9.81*1e-6)
        xf = fv[0]*tf
        yf = fv[1]*tf
        xb = bv[0]*tb
        yb = bv[1]*tb
        XYT[2*i] = np.array([xf, yf, tf])
        XYT[2*i+1] = np.array([xb, yb, tb])
        species[2*i] = 0
        species[2*i+1] = 1

    # first, eliminate any which fall outside the detector bounds
    detector_radius = 40000  # 40 mm in um
    distances = np.sqrt(XYT[:, 0]**2 + XYT[:, 1]**2)
    detected = distances < detector_radius
    XYT_detected = XYT[detected]
    species = species[detected]

    is_f = species == 0
    is_b = species == 1
    fermionsDetected = XYT_detected[is_f].shape[0]
    bosonsDetected = XYT_detected[is_b].shape[0]
    fermionDetection = fermionsDetected/collisions
    bosonDetection = bosonsDetected/collisions

    # now randomly get rid of 75% due to detector inefficiency
    keep_mask = np.random.uniform(0, 1, size=XYT_detected.shape[0]) > 0.75
    XYT_detected = XYT_detected[keep_mask]
    species_detected = species[keep_mask]
    is_f = species_detected == 0
    is_b = species_detected == 1

    t_all = XYT_detected[:, 2]
    t_min, t_max_data = np.nanmin(t_all), np.nanmax(t_all)

    # plot only the XY plane, with a slider to cut off on arrival time
    fig = plt.figure(figsize=(7, 7))
    ax_xy = fig.add_axes([0.12, 0.15, 0.8, 0.75])

    def draw_xy(t_cutoff):
        ax_xy.clear()
        mask_t = t_all <= t_cutoff
        fmask = is_f & mask_t
        bmask = is_b & mask_t
        ax_xy.scatter(XYT_detected[fmask, 0]*1e-3, XYT_detected[fmask, 1]*1e-3, s=4, alpha=1,
                      color=dullPurple, label=r'$^3$He')
        ax_xy.scatter(XYT_detected[bmask, 0]*1e-3, XYT_detected[bmask, 1]*1e-3, s=4, alpha=1,
                      color=gold, label=r'$^4$He')
        circle = Circle((0, 0), detector_radius*1e-3, fill=False, edgecolor='black',
                        linestyle='--', linewidth=1.5)
        ax_xy.add_patch(circle)
        ax_xy.set_xlabel('x (mm)')
        ax_xy.set_ylabel('y (mm)')
        ax_xy.set_aspect('equal')
        ax_xy.set_xlim(-1.2*detector_radius*1e-3, 1.2*detector_radius*1e-3)
        ax_xy.set_ylim(-1.2*detector_radius*1e-3, 1.2*detector_radius*1e-3)
        ax_xy.legend(loc='upper right')

    draw_xy(t_max_data)

    plt.savefig("detector_image.pdf")
    plt.show()

    return XYT_detected, species_detected, fermionDetection, bosonDetection, collisions


def plotDetectionFraction(T, Nratio):

    def runOne(fx, fy, fz, N):
        N_f = (int)(N // (Nratio + 1))
        N_b = (int)(N - N_f)
        mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, N_f, N_b)
        T_c = evaluateCriticalTemperature(fx, fy, fz, N_b)
        k_f = np.sqrt(2*m_F*mu_f)/hbar
        sampling_rate = 4 * k_f
        distance = 1 / sampling_rate
        size = 100 / k_f
        num = (int)(size // distance)
        x_vals = np.linspace(-size / 2, size / 2, num)
        y_vals = x_vals
        z_vals = x_vals
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        n_f, n_b, _ = sampleDensity(fx, fy, fz, 0, mu_f, mu_b, x, y, z)

        n_k, kx, ky, kz, dkx, dky, dkz = findKspaceDistribution(n_b, x_vals, y_vals, z_vals)

        _, _, collisions, _ = evaluateCollisions(mu_f, mu_b, fx, fy, fz, T * T_c, N_f)
        collisions = int(collisions)
        boson_k_samples = sampleMomenta(n_k, kx, ky, kz, dkx, dky, dkz, collisions)
        fp = sampleFermionPositions(x_vals, y_vals, z_vals, n_f, N_f, collisions)
        fermion_k_samples = sampleFermionMomenta(mu_b, mu_f, T * T_c, fx, fy, fz, fp) / hbar

        XYT = np.zeros((2 * collisions, 3))
        species = np.zeros(2 * collisions, dtype=int)  # 0 = fermion (3He), 1 = boson (4He)

        for i in range(collisions):
            bvi = hbar*boson_k_samples[i]/m_B
            bvi[2] -= np.sqrt(2) * vrec
            bki = bvi*m_B/hbar
            fki = fermion_k_samples[i]
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            phi = np.random.uniform(0, 2 * np.pi)
            n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), costheta])
            fkf, bkf = twobodyscatter(fki, bki, n)
            fv = fkf / m_F * hbar
            bv = bkf / m_B * hbar
            s = -848000  # distance to detector plate
            tf = (fv[2] + np.sqrt(fv[2] ** 2 - 2 * 9.81 * 1e-6 * s)) / (9.81 * 1e-6)
            tb = (bv[2] + np.sqrt(bv[2] ** 2 - 2 * 9.81 * 1e-6 * s)) / (9.81 * 1e-6)
            xf = fv[0] * tf
            yf = fv[1] * tf
            xb = bv[0] * tb
            yb = bv[1] * tb
            XYT[2 * i] = np.array([xf, yf, tf])
            XYT[2 * i + 1] = np.array([xb, yb, tb])
            species[2 * i] = 0
            species[2 * i + 1] = 1

        # eliminate any which fall outside the detector bounds
        distances = np.sqrt(XYT[:, 0] ** 2 + XYT[:, 1] ** 2)
        detected = distances < 40000
        XYT_detected = XYT[detected]
        species_detected = species[detected]

        is_f = species_detected == 0
        is_b = species_detected == 1
        fermionsDetected = XYT_detected[is_f].shape[0]
        bosonsDetected = XYT_detected[is_b].shape[0]
        fermionDetection = fermionsDetected / collisions
        bosonDetection = bosonsDetected / collisions

        # randomly get rid of 75% due to detector inefficiency
        keep_mask = np.random.uniform(0, 1, size=XYT_detected.shape[0]) > 0.75
        XYT_detected = XYT_detected[keep_mask]
        species_detected = species_detected[keep_mask]

        return {
            'XYT_detected': XYT_detected,
            'species_detected': species_detected,
            'fermionDetection': fermionDetection,
            'bosonDetection': bosonDetection,
            'collisions': collisions,
            'fx': fx, 'fy': fy, 'fz': fz,
        }

    results = np.zeros((10, 3, 2, 10))
    N = np.linspace(10000, 200000, 10)
    nTrials = 10
    results = np.full((len(N), 3, 2, nTrials), np.nan)
    traps = [("Tight", tight), ("Weak", weak), ("Ultraweak", ultraweak)]

    for i, Ni in enumerate(N):
        for k, (_, f) in enumerate(traps):
            for j in range(nTrials):
                try:
                    r = runOne(f[0], f[1], f[2], Ni)
                except Exception as e:
                    print(f"N={Ni:.0f}, trap {k}, trial {j} failed: {e}")
                    continue
                results[i, k, 0, j] = r["fermionDetection"]
                results[i, k, 1, j] = r["bosonDetection"]

    combined = (results[:, :, 0, :] + results[:, :, 1, :]) / 2
    mean = np.nanmean(combined, axis=2)
    std = np.nanstd(combined, axis=2, ddof=1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, ((name, _), col) in enumerate(zip(traps, [dullPurple, medGray, gold])):
        ax.errorbar(N, mean[:, k], yerr=std[:, k], color=col, marker="o",
                    markersize=3, capsize=3, elinewidth=1.2, label=name)
    ax.set_xlabel("Total Atom Number")
    ax.set_ylabel("Detection Fraction")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("detectionfraction.pdf", bbox_inches="tight")
    plt.show()

    return results

def plotChemPotentials(N_f, N_b):
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

    for i in range(len(T)):
        mu_f_tight[i], mu_b_tight[i] = evaluateEnergies(tight[0], tight[1], tight[2], T[i]*T_c_tight, N_f, N_b)
        mu_f_weak[i], mu_b_weak[i] = evaluateEnergies(weak[0], weak[1], weak[2], T[i]*T_c_weak, N_f, N_b) 
        mu_f_uweak[i], mu_b_uweak[i] = evaluateEnergies(ultraweak[0], ultraweak[1], ultraweak[2], T[i]*T_c_uweak, N_f, N_b) 
    #normalise to initial value
    mu_f_tight = mu_f_tight/mu_f_tight[0]
    mu_b_tight = mu_b_tight/mu_b_tight[0]
    mu_f_weak = mu_f_weak/mu_f_weak[0]
    mu_b_weak = mu_b_weak/mu_b_weak[0]
    mu_f_uweak = mu_f_uweak/mu_f_uweak[0]
    mu_b_uweak = mu_b_uweak/mu_b_uweak[0]
    fig, ax = plt.subplots(2)
    ax[0].plot(T, mu_f_tight, label="Tight Trap")
    ax[0].plot(T, mu_f_weak, label="Weak Trap")
    ax[0].plot(T, mu_f_uweak, label="Ultraweak Trap")
    ax[0].set_xlabel(r"$T/T_c^*$")
    ax[0].set_ylabel(r"$\mu_F(T) / \mu_F(0)$")
    ax[1].plot(T, mu_b_tight, label="Tight Trap")
    ax[1].plot(T, mu_b_weak, label="Weak Trap")
    ax[1].plot(T, mu_b_uweak, label="Ultraweak Trap")
    ax[1].set_xlabel(r"$T/T_c^*$")
    ax[1].set_ylabel(r"$\mu_B(T) / \mu_B(0)$")
    ax[0].legend(loc='upper right')
    ax[1].legend(loc = 'upper right')
    plt.show()

def plotL2Difference(N_f, N_b):
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
    l2_b_tight = np.zeros_like(T)
    l2_b_weak = np.zeros_like(T)
    l2_b_uweak = np.zeros_like(T)
    l2_f_tight = np.zeros_like(T)
    l2_f_weak = np.zeros_like(T)
    l2_f_uweak = np.zeros_like(T)
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
        l2_b_tight[i] = evaluateL2Norm(n_b_tight, n_b_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])/norm_b_tight
        l2_f_tight[i] = evaluateL2Norm(n_f_tight, n_f_tight0, x_tight[:,0,0], y_tight[0,:,0], z_tight[0,0,:])/norm_f_tight
        l2_b_weak[i] = evaluateL2Norm(n_b_weak, n_b_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])/norm_b_weak
        l2_f_weak[i] = evaluateL2Norm(n_f_weak, n_f_weak0, x_weak[:,0,0], y_weak[0,:,0], z_weak[0,0,:])/norm_f_weak
        l2_b_uweak[i] = evaluateL2Norm(n_b_uweak, n_b_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])/norm_b_uweak
        l2_f_uweak[i] = evaluateL2Norm(n_f_uweak, n_f_uweak0, x_uweak[:,0,0], y_uweak[0,:,0], z_uweak[0,0,:])/norm_f_uweak

        

    plt.plot(T, l2_b_tight)
    plt.plot(T, l2_f_tight)
    plt.plot(T, l2_b_weak)
    plt.plot(T, l2_f_weak)
    plt.plot(T, l2_b_uweak)
    plt.plot(T, l2_f_uweak)
    plt.xlabel(r"$T/T_c^*$")
    plt.ylabel("L2 Difference")
    plt.legend(loc = 'lower left')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    all_vals = np.concatenate([l2_b_tight, l2_f_tight, l2_b_weak, l2_f_weak, l2_b_uweak, l2_f_uweak])
    plt.show()

def plotNormInnerProduct(N_f, N_b):
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

    plt.plot(T, ip_b_tight, label="Tight (bosons)")
    plt.plot(T, ip_f_tight, label="Tight (fermions)")
    plt.plot(T, ip_b_weak, label="Weak (bosons)")
    plt.plot(T, ip_f_weak, label="Weak (fermions)")
    plt.plot(T, ip_b_uweak, label="Ultraweak (bosons)")
    plt.plot(T, ip_f_uweak, label="Ultraweak (fermions)")
    plt.xlabel(r"$T/T_c^*$")
    plt.ylabel("Normalised inner product")
    plt.legend(loc = 'lower left')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    all_vals = np.concatenate([ip_b_tight, ip_f_tight, ip_b_weak, ip_f_weak, ip_b_uweak, ip_f_uweak])
    plt.ylim(np.min(all_vals) * 0.98, 1.01)
    plt.show()


def plotTemperatureDifferences(filename="temperature_differences.json"):
    import json

    with open(filename, "r") as f:
        results = json.load(f)

    T = np.array(results["T"])

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    # --- Normalised inner products ---
    ax = axes[0]
    for trap in ["tight", "weak", "uweak"]:
        ax.plot(T, results["inner_product"]["boson"][trap], label=f"{trap} (bosons)")
        ax.plot(T, results["inner_product"]["fermion"][trap], label=f"{trap} (fermions)", linestyle="--")
    ax.set_xlabel(r"$T/T_c^*$")
    ax.set_ylabel("Overlap")
    ax.legend()
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')

    # --- L2 norms ---
    ax = axes[1]
    for trap in ["tight", "weak", "uweak"]:
        ax.plot(T, results["l2_norm"]["boson"][trap], label=f"{trap} (bosons)")
        ax.plot(T, results["l2_norm"]["fermion"][trap], label=f"{trap} (fermions)", linestyle="--")
    ax.set_xlabel(r"$T/T_c^*$")
    ax.set_ylabel("Normalised L2 difference")
    ax.legend()

    # --- Boson number ratio ---
    ax = axes[2]
    
    for trap in ["tight", "weak", "uweak"]:
        ratio = np.array(results["boson_number_ratio"][trap])
        ratioOfTotal = ratio/(1+ratio)
        ax.plot(T, ratioOfTotal, label=trap)
    ax.set_xlabel(r"$T/T_c^*$")
    ax.set_ylabel(r"$N_{thermal}/N_b$")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plotTemperatureDifferences(filename="temperature_differences.json", prefix=""):
    import json

    with open(filename, "r") as f:
        results = json.load(f)

    T = np.array(results["T"])

    colours = {"tight": dullPurple, "weak": medGray, "uweak": gold}
    names = {"tight": "Tight", "weak": "Weak", "uweak": "Ultraweak"}
    xlabel = r"$T/T_c^*$"

    def makePlot(key, ylabel, outfile, ylim_offset=False):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for trap in ["tight", "weak", "uweak"]:
            ax.plot(T, results[key]["boson"][trap], color=colours[trap],
                    linestyle="-", label=f"{names[trap]} (bosons)")
            ax.plot(T, results[key]["fermion"][trap], color=colours[trap],
                    linestyle="--", label=f"{names[trap]} (fermions)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        if ylim_offset:
            ax.ticklabel_format(useOffset=False, style="plain", axis="y")
        fig.tight_layout()
        fig.savefig(f"{prefix}{outfile}", bbox_inches="tight")
        plt.show()

    # 1) Normalised inner product (overlap)
    makePlot("inner_product", "Overlap", "overlap.pdf", ylim_offset=True)

    # 2) Normalised L2 difference
    makePlot("l2_norm", "Normalised L2 difference", "l2difference.pdf")

    # 3) Thermal fraction of bosons (bosons only, so solid lines)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for trap in ["tight", "weak", "uweak"]:
        ratio = np.array(results["boson_number_ratio"][trap])
        ax.plot(T, ratio / (1 + ratio), color=colours[trap],
                linestyle="-", label=names[trap])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$N_{\mathrm{thermal}}/N_b$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}thermalfraction.pdf", bbox_inches="tight")
    plt.show()

def plotBoseCollisions(N):
    FET = np.linspace(0, 1500, 25)
    tc = np.zeros_like(FET)
    wc = np.zeros_like(FET)
    uc = np.zeros_like(FET)
    for i in range(len(FET)):
        fx, fy, fz = tight[0], tight[1], tight[2]
        mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)
        _, _, tc[i] = boseBoseCollision(mu_b, fx, fy, fz, FET[i])
        fx, fy, fz = weak[0], weak[1], weak[2]
        mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)
        _, _, wc[i] = boseBoseCollision(mu_b, fx, fy, fz, FET[i])
        fx, fy, fz = ultraweak[0], ultraweak[1], ultraweak[2]
        mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)
        _, _, uc[i] = boseBoseCollision(mu_b, fx, fy, fz, FET[i])
    plt.scatter(FET, tc, label="Tight", color=dullPurple)
    plt.scatter(FET, wc, label="Weak", color=medGray)
    plt.scatter(FET, uc, label="Ultraweak", color=gold)
    plt.legend(loc="upper right")
    plt.xlabel(rf"Free expansion time ($\mu s$)")
    plt.ylabel("Collision Number")
    plt.savefig("collisionsvsexpansiontime.pdf")
    plt.show()

def plot_scatters(*files):
    """
    Plot scatters vs atom_number from multiple CSV files.

    Parameters
    ----------
    *files : str
        Paths to CSV files. Each file should contain columns
        'scatters' and 'atom_number'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for file in files:
        df = pd.read_csv(file)

        ax.scatter(
            df['atom_number'],
            df['scatters'],
            label=file,
            alpha=0.7
        )

    ax.set_xlabel('Atom Number')
    ax.set_ylabel('Scatters')
    ax.set_title('Scatters vs Atom Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_median_scatters(
    *files,
    bin_width=500,
    atom_col='atom_number',
    scatter_col='scatters'
):
    """
    Plot median scatters against binned atom numbers
    for multiple CSV files.

    Parameters
    ----------
    *files : str
        Paths to CSV files, one per trap.
    bin_width : float
        Width of atom-number bins.
    atom_col : str
        Column containing atom numbers.
    scatter_col : str
        Column containing the number of scatters.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for file in files:
        df = pd.read_csv(file)

        # Remove rows with missing data
        df = df.dropna(subset=[atom_col, scatter_col])

        # Assign each point to an atom-number bin
        df['atom_bin'] = (
            np.floor(df[atom_col] / bin_width) * bin_width
        )

        # Calculate median scatters in each bin
        grouped = df.groupby('atom_bin')[scatter_col].median()

        # Plot medians at bin centres
        bin_centres = grouped.index + bin_width / 2

        ax.scatter(
            bin_centres,
            grouped.values,
            label=file,
            alpha=0.8
        )

    ax.set_xlabel('Atom Number')
    ax.set_ylabel('Median Number of Scatters')
    ax.set_title('Median Scatters vs Binned Atom Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

evaluateDetectorImageXY(tight[0], tight[1], tight[2], 0.6, 40000, 3)