#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (ymniquet@gmail.com).
# Version: 2023.03

from pylab import *
from fdint import * # Fermi integrals package fdint.
from constants import me, mp, ec, hbar, kb

"""Plot density of states and Fermi-Dirac distribution for a fermion gas."""

###################
# Gaz parameters. #
###################

# For an electron gas :

m = me # Particle mass (kg).
nd = 2 # Spin 1/2 degeneracy.

###################

### Functions.

def chemical_potential(n, T):
  """Returns chemical potential (eV) at given temperature T (K) and density n (m^{-3})."""
  kT = kb*T
  pn = (nd/(2*pi)**2)*(2*m/hbar**2)**1.5*kT**1.5 # Prefactor for the density of particles.
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
  return rmu*kT
  
def plot_DoS(n, T, c = "b", label = ""):
  """Plot density of states and Fermi-Dirac distribution at given temperature T (K) and density n (m^{-3}), with color c and label label."""
  kT = kb*T    
  mu = chemical_potential(n, T)
  Es = linspace(min(0, mu*kT), mu+30*kT, 1024)
  rhos = (nd/(2*pi)**2)*(2*m/hbar**2)**1.5*sqrt(maximum(0, Es))
  rhofs = rhos/(1+exp((Es-mu)/kT))
  plot(Es/ec, rhos *ec/1e6, "k-") # Plot energies in eV, densities in cm^-3.
  plot(Es/ec, rhofs*ec/1e6, "-", color = c, label = label)
  fill_between(Es/ec, rhofs*ec/1e6, color = c, alpha = .5)
  axvline(mu/ec, linestyle = "-.", color = c)
  print(f"DoS integral = {sum(rhofs)*(Es[1]-Es[0]):.2e}/m^3 (shall be {n:.2e}/m^3).")
    
### Plots.

n = 6.5e31 # Density of electrons in the core of the Sun (m^-3).
T = 15e6 # Temperature in the core of the Sun (K).

figure()
plot_DoS(n, T, "b", label = "$T=1.5\\times 10^7$\,K")
xlabel("$E$ (eV)")
ylabel("$\\rho$ (eV$^{-1}$.cm$^{-3}$)")
title("$n=6.5\\times 10^{25}$\,cm$^{-3}$")
legend(loc = "lower right")
xlim(-2000, 8000)
ylim(0, .6e24)
text(0.025, 0.965, "(a)", fontsize = 20, ha = "left", va = "top", transform = gca().transAxes)

savefig("classique.pdf")

n = 3e35 # Density of electrons in a white dwarf (m^-3).
T = 1e7 # Temperature of a white dwarf (K).

figure()
plot_DoS(n, T, "b", label = "$T=10^7$\,K")
xlabel("$E$ (eV)")
ylabel("$\\rho$ (eV$^{-1}$.cm$^{-3}$)")
title("$n=3\\times 10^{29}$\,cm$^{-3}$")
legend(loc = "lower center")
xlim(0, 1.75e5)
ylim(0, 3e24)
text(0.025, 0.965, "(b)", fontsize = 20, ha = "left", va = "top", transform = gca().transAxes)

savefig("degenere.pdf")

show()


