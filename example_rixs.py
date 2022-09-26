#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""template."""

# %% imports ===================================================================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import copy

import spectragen4 as sg4
import brixs as br

%matplotlib qt5
%load_ext autoreload
%autoreload 2


# %% Initial calculation paramenters ===========================================
q = sg4.Calculation(element='Cu', charge='2+', symmetry='D4h', experiment='RIXS', edge='L2,3-M4,5 (2p3d)', toCalculate='ld')
q.xMin = 932.7-0.1
q.xMax = 932.7+0.1
q.xNPoints = 20
q.yMin = -0.5
q.yMax = 2

Dq = 0.09
Ds = -0.07
for h in ['Initial Hamiltonian', 'Intermediate Hamiltonian', 'Final Hamiltonian']:
    q.hamiltonianData['Crystal Field'][h]['Dq(3d)'] = Dq
    q.hamiltonianData['Crystal Field'][h]['Ds(3d)'] = Ds
    q.hamiltonianData['Crystal Field'][h]['Dt(3d)'] = -(0.45+4*Ds)/5#-(0.45+4*Ds)/5
# for h in ['Initial Hamiltonian', 'Intermediate Hamiltonian', 'Final Hamiltonian']:
#     q.hamiltonianData['Atomic'][h]['ζ(3d)'][1] = 2
print(q.hamiltonianData)
q.save_parameters('parameters_z2')

Dq = 0.12
Ds = 0.08
for h in ['Initial Hamiltonian', 'Intermediate Hamiltonian', 'Final Hamiltonian']:
    q.hamiltonianData['Crystal Field'][h]['Dq(3d)'] = Dq
    q.hamiltonianData['Crystal Field'][h]['Ds(3d)'] = Ds
    q.hamiltonianData['Crystal Field'][h]['Dt(3d)'] = (0.45-4*Ds)/5#(0.45-4*Ds)/5
q.save_parameters('parameters_x2-y2')
# print(q.hamiltonianData)
# q.hamiltonianState

# %% Energy map template =======================================================
q = sg4.Calculation()
q.load_parameters('parameters_z2.par')

q.save_input()
output = q.run()
ss, _ = q.spectrum()

# plot spectra
plt.figure()
_ = ss.plot()
plt.xlabel('Energy loss (eV)')
plt.ylabel('Intensity (arb. units)')

# plot map
map = ss.calculate_map(axis='h')
map.y_centers = ss.incident_energy

plt.figure()
map.plot()
plt.xlabel('Energy loss (eV)')
plt.ylabel('Excitation energy (eV)')

# get resonance energy
i_max = 0
for i in range(len(ss.incident_energy)):
    if max(ss[i].y) > max(ss[i_max].y):
        i_max = i
print(i_max)
print(ss[i_max].incident_energy)

# plt.close('all')





# %%
