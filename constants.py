#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (ymniquet@gmail.com).
# Version: 2023.09

import math

"""Physical constants."""

me = 9.1093837e-31 # Electron mass (kg).
ec = 1.602176634e-19 # Electron charge (C).
mp = 1.67262192e-27 # Proton mass (kg).
h = 6.62607015e-34 # Planck constant (J.s).
hbar = h/(2.*math.pi) # Rationalized Planck constant (J.s).
Na = 6.02214076e23 # Avogadro number.
kb = 1.380649e-23 # Boltzmann constant (J/K).
R = Na*kb # Ideal gas constant (J/mol/K).
c = 299792458 # Light velocity (m/s).
sigma = 2.*math.pi**5*kb**4/(15.*h**3*c**2) # Stefan-Boltzmann constant (W/m^2/K^4).
G = 6.6743e-11 # Gravitational constant (m^3/kg/s^2).
atm = 101325. # Atmosphere pressure (Pa).
ge = 9.80665 # Gravitational field on Earth (m/s^2).
