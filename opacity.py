#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2023.09

from pylab import *
import star as star

"""Plot star opacity for different models."""

###############
# Parameters. #
###############

X = 0.7346 # Hydrogen mass fraction.
Z = 0.0169 # Metallicity.

Ts = 10**linspace(3.75, 8., 100) # Temperature grid (K).
rhos = [1.e-4, 1.e-2, 1.e0, 1.e2, 1.e4] # Mass densities (kg/m^3).
labels = ["$10^{-4}$", "$10^{-2}$", "$10^{0}$", "$10^{2}$", "$10^{4}$"] # Label of each mass density.
colors = ["cyan", "blue", "magenta", "orange", "red"] # Color of each mass density.

###################

### Functions.

def plot(model, linestyle = "-", add_labels = True):
  """Plot opacity for a given model."""
  for rho, color, label in zip(rhos, colors, labels):
    loglog(Ts, model.kappa(rho, Ts), linestyle = linestyle, color = color, label = label if add_labels else None)

### Plots.

figure(1)

# Plot Kramer's opacity.

#Kramers = star.KramersHydrostaticMainSequenceStar(1.99e30, 2, 1.)
Kramers = star.ModKramersHydrostaticMainSequenceStar(1.99e30, 2, 1.)
Kramers.set_composition(X, Z)

plot(Kramers, "--", add_labels = False)

# Plot OPAL opacity.

OPAL = star.OPALHydrostaticMainSequenceStar(1.99e30, 2, 1.)
OPAL.set_composition(X, Z)

plot(OPAL, "-")

# Finalize plot.

xlabel("$T$ (K)")
ylabel("$\kappa$ (m$^2$/kg)")
legend(loc = "upper right", fontsize = 14, title = "$\\rho$ (kg/m$^3$)")
title(f"$X={X:.4}$, $Z={Z:.4}$", fontsize = 16)
ylim(1.e-2, 1.e5)
grid()

savefig("opacite.pdf")

show()
