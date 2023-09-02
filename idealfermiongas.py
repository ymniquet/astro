#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2023.09

from pylab import *
try:
  from fdint import * # Fermi integrals package fdint.
except:
  raise RuntimeError("Please install the fdint package with 'pip install --user fdint'.")
from constants import me, mp, hbar, kb

"""Plot fermion gas pressure versus density & temperature."""

###################
# Gaz parameters. #
###################

# For an electron gas :

m = me # Particle mass (kg).
nd = 2 # Spin 1/2 degeneracy.

###################

### Functions.

def pressure(T, n):
  """Returns pressure (Pa) at given temperature T (K) and density n (m^{-3})."""
  kT = kb*T
  pn = (nd/(2*pi)**2)*(2*m/hbar**2)**1.5*kT**1.5 # Prefactor for the density of particles.
  pe = (nd/(2*pi)**2)*(2*m/hbar**2)**1.5*kT**2.5 # Prefactor for the density of energy.
  # Find the reduced chemical potential rmu = mu/kT using a Newton-Raphson algorithm.
  nit = 0
  rmu = log(n/pn) # Start from the degenerate gas solution.
  while True:
    nrmu = pn*fdk(0.5, rmu)
    deltan = nrmu-n
    if abs(deltan)/n < 1e-9: break # Converged.
    nit += 1
    if nit > 1000: raise RuntimeError("Failed to solve the chemical potential.")
    dnrmu = pn*dfdk(0.5, rmu) # Newton-Raphson correction.
    rmu = rmu-deltan/dnrmu
  e = pe*fdk(1.5, rmu) # Density of energy.
  P = 2*e/3 # Pressure.
  return P

def plot_PT(ax, n, Ts, c = "b"):
  """Plot pressure in axes ax as a function of temperatures Ts (K) for a given density n (m^{-3}), with color c."""
  l = 1e10*n**(-1./3) # Average inter-particle distance (A).
  Ps = array([pressure(T, n) for T in Ts])
  ax.loglog(Ts, Ps, "-", color = c, lw = 2.5, label = f"$\Lambda={l:.4G}$\,\AA")
  ax.plot(Ts, n*kb*Ts, ":", color = c) # Classical, non-degenerate limit.
  Tdeg = (2*pi*hbar**2/(m*kb))*(n/nd)**(2./3) # Degeneracy temperature.
  ax.axvline(Tdeg, linestyle = "-.", color = c)
  Pdeg = (hbar**2/(5*m))*(6*pi**2/nd)**(2./3)*n**(5./3) # Degenerate pressure.
  ax.axhline(Pdeg, linestyle = "--", color = c)

def plot_Pn(ax, T, ns, c = "b"):
  """Plot pressure in axes ax as a function of densities ns (m^{-3}) for a given temperature T (K), with color c."""
  ls = 1e10*ns**(-1./3) # Average inter-particle distance (A).
  Ps = array([pressure(T, n) for n in ns])
  ax.loglog(ls, Ps, "-", color = c, lw = 2.5, label = f"$T=10^{log10(T):.0f}$\,K")
  ax.plot(ls, ns*kb*T, ":", color = c) # Classical, non-degenerate limit.
  ndeg = nd*(m*kb*T/(2*pi*hbar**2))**1.5 # Degeneracy density.
  ax.axvline(1e10*ndeg**(-1./3), linestyle = "-.", color = c)

### Plots.

cs = ["b", "m", "r", "g"] # Plot colors.

# Plot pressure as a function of T for various densities.

Ts = 10**linspace(2, 10, 256) # Temperatures (K).

ls = array([.1, 1, 10, 100]) # Average inter-particle distances (A).
ns = 1/(ls*1e-10)**3 # Densities (m^{-3}).

figure(layout = "constrained")
ax = axes()
for n, c in zip(ns, cs): plot_PT(ax, n, Ts, c)
ax.legend(loc = "lower right", ncol = 2, fontsize = 14)
ax.set_xlabel("$T$ (K)")
ax.set_ylabel("$P$ (Pa)")
ax.set_xlim(1e2, 1e10)
ax.set_ylim(1e2, 1e20)
ax.text(0.025, 0.965, "(a)", fontsize = 20, ha = "left", va = "top", transform = ax.transAxes)

savefig("P_T.pdf")

# Plot pressure as a function of density for various T.

Ts = [1e2, 1e4, 1e6, 1e8] # Temperatures (K).

ls = 10**linspace(-1, 2, 256) # Average inter-particle distances (A).
ns = 1/(ls*1e-10)**3 # Densities (m^{-3}).

figure(layout = "constrained")
axl = axes()
for T, c in zip(Ts, cs): plot_Pn(axl, T, ns, c)
Pdegs = (hbar**2/(5*m))*(6*pi**2/nd)**(2./3)*ns**(5./3) # Degenerate limit.
axl.plot(ls, Pdegs, "--", color = "black")
axl.legend(loc = "lower left", ncol = 2, fontsize = 14)
axl.set_xlabel("$\Lambda$ (\AA)")
axl.set_ylabel("$P$ (Pa)")
axl.set_xlim(1e-1, 1e2)
axl.set_ylim(1e2, 1e20)
axl.text(0.975, 0.965, "(b)", fontsize = 20, ha = "right", va = "top", transform = axl.transAxes)
# Add density axis.
axr = axl.twiny()
axr.set_xscale("log")
axr.set_xlim(axl.get_xlim())
lticks = array([1e-1, 1e0, 1e1, 1e2])
rticks = 1/(lticks*1e-8)**3
axl.set_xticks(lticks)
axr.set_xticks(lticks)
axr.set_xticklabels([f"$10^{{{log10(r):.0f}}}$" for r in rticks])
axr.set_xlabel("$n$ (cm$^{-3}$)")

savefig("P_n.pdf")

show()

