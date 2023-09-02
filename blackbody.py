#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (ymniquet@gmail.com).
# Version: 2023.09

from pylab import *
from constants import h, hbar, kb, c

"""Plot black-body spectrum."""

###############
# Parameters. #
###############

T = 5775 # Temperature (K).
R = 696340e3 # Sun diameter (m).
d = 150e9 # Sun-Earth distance (m).

###################

### Plot spectral luminance.

kT = kb*T

l = linspace(1e-9, 6000e-9, 5999) # Wavelength (m).
f = c/l # Frequency (Hz).
Phi = ((2*pi*h*f**3/c**2)/(exp(h*f/kT)-1))*(R/d)**2/(h*l) # Flux of photons per second, per m^2 of Earth and per m of wavelength.

print(f"Irradiance = {sum(Phi*f)*h*(l[1]-l[0]):.2f} W/m^2 (shall be 1360 W/m^2 for full spectrum integration).")

figure(layout = "constrained")
axl = axes()
axl.plot(l*1e9, Phi/1e27, "b-")
axl.set_xlabel("$\lambda$ (nm)")
axl.set_ylabel("$\Phi_\lambda$ ($10^{18}$ photons/s/m$^2$/nm)")
axl.set_xlim(0, 2000)
axl.set_ylim(0, 6)
# Add frequency axis.
axf = axl.twiny()
axf.set_xlim(axl.get_xlim())
lticks = array([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
fticks = c/(lticks[1:]*1.e-9)/1e12
axl.set_xticks(lticks)
axf.set_xticks(lticks[1:])
axf.set_xticklabels([f"{f:.0f}" for f in fticks])
axf.set_xlabel("$\\nu$ (THz)")
# Plot visible light range.
gradient = linspace(0, 1, 256)
gradient = vstack((gradient, gradient))
axl.imshow(gradient, extent = [400, 700, 0, 6], cmap = colormaps["rainbow"], aspect = "auto", alpha = .5)

savefig("corpsnoir.pdf")

show()
