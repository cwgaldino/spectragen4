#! /usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Package for spectrum generation via crispy 0.7.4 and Quanty.

Usage

1) Download the latest version (right now, this is 0.7.4 - 2024-05-27) from GitHub
https://github.com/mretegan/crispy, unzip it.

2) Download Quanty for Windows https://www.quanty.org/index.html

3) Import this file in your python script

>>> import sys
>>> sys.path.append(r'<path-to-spectragen>\spectragen4')
>>> import spectragen4 as sg4

4) set quanty and crispy filepaths

>>> sg4.settings.QUANTY_FILEPATH = r'C:\Users\galdin_c\github\quanty\quanty_win\QuantyWin64.exe'
>>> sg4.settings.CRISPY_FILEPATH = r'C:\Users\galdin_c\github\crispy-0.7.4'

5) if you want to use it with brixs, set USE_BRIXS to True

>>> sg4.settings.USE_BRIXS = True

6) basic calculation

TO DO





7) Notes

For a octahedral environment, note that the crystal field coordinate system is
such that the planar ligands point to the x and y coordinates and the apical 
ligands are laid along the z axis. Therefore, one can set k1, eps11, and eps12
such as the example below:

Top view (lab coordinates)
                       .
                     .   .
LH y (010)         .       .
    |            .           .
────O────────  .   octahedron  . ────> x (100)
 LV z (001       .           .
                   .       .
                     .   .
                       .

octahedron z: points out of the screen
lab z: points out of the screen




TO DO:
( ) Check if broaden works with non-monotonic data.
( ) Not sure if broaden works fine. Compare with crispy.
"""

# %% Imports ===================================================================
# standard libraries
from pathlib import Path
import numpy as np
import copy
import sys
import os
import subprocess

# specific libraries
from collections.abc import MutableMapping
import tempfile
import shutil
import gzip
import json

# %% broadening functions ======================================================
def _broaden(self, value):
    """Apply gaussian broadening to spectrum.

    Args:
        value (number): fwhm value of the gaussian broadening.

    Returns:
        None
    """
    # xScale = np.abs(self.x.min() - self.x.max()) / self.x.shape[0]
    # fwhm = float(value)/float(xScale)
    s = self.copy()
    s.y = broaden(self.x, self.y, value)
    return s

def _broaden2(self, value):
    """Apply gaussian broadening to spectrum.

    Args:
        value (number): fwhm value of the gaussian broadening.

    Returns:
        None
    """
    # xScale = np.abs(self.x.min() - self.x.max()) / self.x.shape[0]
    # fwhm = float(value)/float(xScale)
    self.y = broaden(self.x, self.y, value)

def broaden(x, y, value):
    """Apply gaussian broadening to spectrum.

    Args:
        value (number): fwhm value of the gaussian broadening.

    Returns:
        Broadened y
    """
    fwhm = value
    xScale = np.abs(min(x) - max(x)) / x.shape[0]
    fwhm = float(value)/float(xScale)
    y = settings.broaden1(y, fwhm, 'gaussian')  # broaden1 is loaded with crispy down below
    return y

# %% support functions =========================================================
def _normalize(vector):
    """Returns normalized vector."""
    if np.linalg.norm(vector) != 1:
        return vector/np.linalg.norm(vector)
    else:
        return vector


# The functions down below are used to print/save the initial wavefunctions 
# specifically for Cu2+ (d9). It will not work for other ions because the base
# wavefunctions are just too many due to the ion having more than one electron 
# hole in the d shell. Although possible, I think it is annoying to write the 
# base wave functions down in terms of spherical harmonics. Also, I don't see the
# real benefit of it now. So, as for right now, I am not even thinking 
# about expanding this to "multi-electronic" ions.

# one thing that one could do easily, would be to expand this code to other d9 
# ions other than Cu2+. For that one would just have to check the number of electrons
# or holes in the the d shell before printing/saving the initial wave functions.

# I only tested this code for Oh and D4h geometry, but I fail to see any reason
# why it wouldn't work for any other geometry.

text_defining_functions_for_print_wavefunction_for_d4h_3d_ions = r"""--------------------------------------------------------------------------------
-- functions for saving/printing initial wavefunctions (only works for 3d metals in an Oh/D4h env.)
--------------------------------------------------------------------------------
-- z2    = Y^0
-- x2-y2 = 1/sqrt(2) [Y-2 + Y2]
-- xy    = 1/i*sqrt(2) [Y-2 - Y2]
-- yz    = i/sqrt(2) [Y-1 + Y1]
-- xz    = 1/sqrt(2) [Y-1 - Y1]

function replace_char(pos, str, r)
    return str:sub(1, pos-1) .. r .. str:sub(pos+1)
end


function print_psi(Psis_i, base_psis, base_dets, base_sphe, H_i, filepath, verbose)

    -- wavefunction name and energy
    for i, psi in ipairs(Psis_i) do
        number_of_eigenfunctions = i
    end
    
    -- initial header
    text = ''
    text = text .. '#########################\n'
    text = text .. '# INITIAL WAVEFUNCTIONS #\n'
    text = text .. '#########################\n'
    text = text .. '# Number of wavefunctions: ' .. number_of_eigenfunctions .. '\n'
    text = text .. '# number Energy wavefunction\n'
    for i, psi in ipairs(Psis_i) do
        text = text .. '    ' .. i .. '      <<E' .. i .. '>>  <<psi' .. i .. '>>' .. '\n'
    end
    text = text .. '# \n'

    for i, psi in ipairs(Psis_i) do
        -- calculate energy
        if H_i ~= nil then
            _E = Complex.Re(psi * H_i * psi) -- Taking the real part only because sometimes the precision error creates a small im component
        else
            _E =  0
        end

        -- wavefunction name and energy
        text = text .. '###################\n'
        text = text .. '# WAVEFUNCTION: ' .. i ..' # \n'
        text = text .. '###################\n'
        text = text .. '# E' .. i .. ' = ' .. _E.. '\n'
        -- text = text .. '# \n'

        -- M holes
        text = text .. '# =============== M holes ===============\n'
        if H_3d_ligands_hybridization_lmct == 1 then
            text = text .. '# 2p       3d         L     Y_2^ml(ms) factor\n'
        else
            text = text .. '# 2p       3d    Y_2^ml(ms) factor\n'
        end
        for j = 1, 10, 1 do
            factor = psi*base_psis[j]
            if math.abs(Complex.Im(factor)) > epsilon then
                _factor = '(' .. Complex.Re(factor) .. ', ' .. Complex.Im(factor) .. '*I)'
                text = text .. base_dets[j]  .. '  ' .. base_sphe[j] .. '    ' .. _factor .. '\n'
            else
                text = text .. base_dets[j]  .. '  ' .. base_sphe[j] .. '    ' .. Complex.Re(factor) .. '\n'
            end
        end

        -- find wavefunction with highest factor
        -- highest = psi*base_psis[1]
        factor = psi*base_psis[1]
        highest = Complex.Re(factor)^2 + Complex.Im(factor)^2
        for j = 2, 10, 1 do
            _factor = psi*base_psis[j]
            _temp = Complex.Re(_factor)^2 + Complex.Im(_factor)^2
            if _temp > highest then
                highest = _temp
            end
        end

        -- find all wavefunctions where contribution is larger than 10% of highest contribution
        _final = 'psi' .. i .. ' = '
        for j = 1, 10, 1 do 
            _factor = psi*base_psis[j]
            _temp = Complex.Re(_factor)^2 + Complex.Im(_factor)^2
            if math.abs(_temp) > math.abs(highest)*0.1 then
                factor = psi*base_psis[j]
                if math.abs(Complex.Im(_factor)) > epsilon then
                    _factor = '(' .. tonumber(string.format("%.3f", math.abs(Complex.Re(factor)))) .. ', ' .. tonumber(string.format("%.3f", math.abs(Complex.Im(factor))))  .. '*I)'
                    _final = _final .. ' + ' .. _factor .. '*' .. base_sphe[j]
                else
                    _factor = tonumber(string.format("%.3f", math.abs(Complex.Re(factor))))
                    if Complex.Re(factor) < 0 then
                        _final = _final .. ' - ' .. _factor .. '*' .. base_sphe[j]
                    else
                        _final = _final .. ' + ' .. _factor .. '*' .. base_sphe[j]
                    end
                end

            end
        end
        -- _final = string.sub(_final, 1, -3)

        -- print final psi_i
        text = text .. '# \n'
        text = text .. '# Estimated wavefunction in terms of spherical harmonics:\n'
        text = text .. _final .. '\n'
        text = text .. '# \n'

        -- L holes
        if H_3d_ligands_hybridization_lmct == 1 or H_3d_ligands_hybridization_mlct == 1 then
            text = text .. '# =============== L holes ===============\n'
            text = text .. '# 2p       3d         L    factor\n'
            for j = 11, 20, 1 do
                factor = psi*base_psis[j]
                if math.abs(Complex.Im(factor)) > epsilon then
                    _factor = '(' .. Complex.Re(factor) .. ', ' .. Complex.Im(factor) .. '*I)'
                    text = text .. base_dets[j]  .. '  ' .. _factor .. '\n'
                else
                    text = text .. base_dets[j]  .. '  '  .. Complex.Re(factor) .. '\n'
                end
                -- text = text .. base_dets[j] .. ' ' .. psi*base_psis[j] .. '\n'
            end
        end

        -- end of wavefunction table
        text = text .. '# \n'

        -- update header
        text = string.gsub(text, '<<psi' .. i .. '>>', _final)
        text = string.gsub(text, '<<E' .. i .. '>>', _E)

    end

    if verbose == 1 then
        print(text)
    end

    if filepath ~= nil then
        file = io.open(filepath, "w")
        file:write(text)
        file:close()
    end
end
"""

text_defining_base_for_printing_wavefunction_for_d4h_3d_ions = r"""-- base wavefunctions for printing/saving initial wavefunctions (only works for 3d metals in an Oh/D4h env.)
--------------------------------------------------------------------------------
-- z2    = Y^0
-- x2-y2 = 1/sqrt(2) [Y-2 + Y2]
-- xy    = 1/i*sqrt(2) [Y-2 - Y2]
-- yz    = i/sqrt(2) [Y-1 + Y1]
-- xz    = 1/sqrt(2) [Y-1 - Y1]

base_psis = {}
base_dets = {}
base_sphe = {'Y^-2(-)', 'Y^-2(+)', 'Y^-1(-)', 'Y^-1(+)', 'Y^0(-)', 'Y^0(+)', 'Y^1(-)', 'Y^1(+)', 'Y^2(-)', 'Y^2(+)'}
if H_3d_ligands_hybridization_lmct == 1 or H_3d_ligands_hybridization_mlct == 1 then
    for i = 1, 20, 1 do
        _det  = replace_char(6+i, '11111111111111111111111111', '0') 
        _psi = NewWavefunction(NFermions, NBosons, {{_det, 1}})
        table.insert(base_psis, _psi)

        _det2 = replace_char(6, _det, '1 ') 
        if i < 11 then
            _det2 = replace_char(18, _det2, ' 1') 
        else
            _det2 = replace_char(17, _det2, '1 ') 
        end
        table.insert(base_dets, _det2)
    end
else
    for i = 1, 10, 1 do
        _det  = replace_char(6+i, '1111111111111111', '0') 
        _psi = NewWavefunction(NFermions, NBosons, {{_det, 1}})
        table.insert(base_psis, _psi)

        _det2 = replace_char(6, _det, '1 ') 
        table.insert(base_dets, _det2)
    end
end

-- z2    = Y^0
psi_z2_m    = base_psis[5]
psi_z2_p    = base_psis[6]

-- x2-y2 = 1/sqrt(2) [Y-2 + Y2]
psi_x2_y2_m = math.sqrt(1/2)*base_psis[1] + math.sqrt(1/2)*base_psis[9]
psi_x2_y2_p = math.sqrt(1/2)*base_psis[2] + math.sqrt(1/2)*base_psis[10]

-- xy    = 1/i*sqrt(2) [Y-2 - Y2]
psi_xy_m = (1/I)*math.sqrt(1/2)*base_psis[1] - (1/I)*math.sqrt(1/2)*base_psis[9]
psi_xy_p = (1/I)*math.sqrt(1/2)*base_psis[2] - (1/I)*math.sqrt(1/2)*base_psis[10]

-- yz    = i/sqrt(2) [Y-1 + Y1]
psi_yz_m = I*math.sqrt(1/2)*base_psis[3] + I*math.sqrt(1/2)*base_psis[7]
psi_yz_p = I*math.sqrt(1/2)*base_psis[4] + I*math.sqrt(1/2)*base_psis[8]

-- xz    = 1/sqrt(2) [Y-1 - Y1]
psi_xz_m = math.sqrt(1/2)*base_psis[3] - math.sqrt(1/2)*base_psis[7]
psi_xz_p = math.sqrt(1/2)*base_psis[4] - math.sqrt(1/2)*base_psis[8]

"""


# %% settings ==================================================================
class _settings():

    def __init__(self):
        self._QUANTY_FILEPATH   = ''
        self._CRISPY_FILEPATH   = ''
        self._CRISPY_PATH_INDEX = 0
        self.default    = ''
        self._USE_BRIXS = False

    @property
    def QUANTY_FILEPATH(self):
        return self._QUANTY_FILEPATH
    @QUANTY_FILEPATH.setter
    def QUANTY_FILEPATH(self, value):
        value = Path(value)
        assert value.exists(), 'Cannot find filepath'
        assert value.is_file(), 'filepath does not point to a file'
        self._QUANTY_FILEPATH = value
    @QUANTY_FILEPATH.deleter
    def QUANTY_FILEPATH(self):
        raise AttributeError('Cannot delete object.')
    
    @property
    def CRISPY_FILEPATH(self):
        return self._CRISPY_FILEPATH
    @CRISPY_FILEPATH.setter
    def CRISPY_FILEPATH(self, value):
        # inside <folderpath2crispy> one must find the folders 'crispy', 'docs', 'scripts', etc...
        value = Path(value)
        assert value.exists(), 'Cannot find filepath'
        assert value.is_dir(), 'filepath does not point to a folder'

        # delete previously loaded functions
        try:
            del QuantyCalculation
            del broaden1
            sys.path.pop(self._CRISPY_PATH_INDEX)
        except NameError:
            pass

        # load crispy functions
        self._CRISPY_PATH_INDEX = len(sys.path)
        sys.path.append(str(value))
        from crispy.gui.quanty import QuantyCalculation
        from crispy.gui.quanty import broaden as broaden1

        # load crispy settings.default parameters
        f = gzip.open(value/r'crispy\modules\quanty\parameters\parameters.json.gz', 'r')
        temp = json.load(f)
        self.default = temp['elements']
        del temp

        # save
        self.QuantyCalculation = QuantyCalculation
        self.broaden1          = broaden1
        self._CRISPY_FILEPATH  = value
    @CRISPY_FILEPATH.deleter
    def CRISPY_FILEPATH(self):
        raise AttributeError('Cannot delete object.')
    
    @property
    def USE_BRIXS(self):
        return self._USE_BRIXS
    @USE_BRIXS.setter
    def USE_BRIXS(self, value):
        assert isinstance(value, bool), 'USE_BRIXS must be True or False'
        if value:
            import brixs as _br
            settings._br = _br
            settings._br.Spectrum.broaden = _broaden
            settings._br.Spectrum.broaden2 = _broaden2
        # save
        self._USE_BRIXS = value
    @USE_BRIXS.deleter
    def USE_BRIXS(self):
        raise AttributeError('Cannot delete object.')
settings = _settings()


# %% Operating system ==========================================================
import platform
system = platform.system().lower()
is_windows = system == 'windows'
is_linux   = system == 'linux'
is_mac     = system == 'darwin'


# %% support classes ===========================================================
class _hamiltonianState(MutableMapping):
    def __init__(self, initial, parent):
        self.store = dict(initial)
        self.parent = parent

        # change from int to bool
        for name in self.store:
            if self.store[name]:
                self.store[name] = True
            else:
                self.store[name] = False

    def __str__(self):
        return str(self.store).replace(', ', '\n ')

    def __repr__(self):
        # return str(self.store)
        return str(self.store).replace(', ', '\n ')

    def __getitem__(self, name):
        return self.store[name]

    def __setitem__(self, name, value):
        if name not in self:
            raise ValueError(f'New terms cannot be created.\nInvalid term: {name}\nValid terms are: {list(self.keys())}')
        assert isinstance(value, bool), f'hamiltonianState[{name}] must be a bool (True or False), not type: {type(value)}'

        if self.store[name] == False and value == True:  # from False to True
            
            # check if LMCT and MLCT are True at the same time
            if name == '3d-Ligands Hybridization (LMCT)':
                if self.store['3d-Ligands Hybridization (MLCT)']:
                    raise ValueError('Ligands Hybridization LMCT and MLCT cannot be both True.\nIf you want to set LMCT to True, you have to set MLCT to False first')
            elif name == '3d-Ligands Hybridization (MLCT)':
                if self.store['3d-Ligands Hybridization (LMCT)']:
                    raise ValueError('Ligands Hybridization MLCT and LMCT cannot be both True.\nIf you want to set MLCT to True, you have to set LMCT to False first')
            
            # set value
            self.store[name] = True

            # Determine the maximum number of allowed configurations.
            if name == '3d-Ligands Hybridization (LMCT)':
                if 'd' in self.parent.q.block:
                    self.parent.q.nConfigurationsMax = 10 - self.parent.nElectrons + 1
                elif 'f' in self.parent.q.block:
                    self.parent.q.nConfigurationsMax = 14 - self.parent.nElectrons + 1
                self.parent.q.nConfigurations = 2
                print(f'LMCT changed from False to True: nConfigurationsMax changed from 1 to {self.parent.nConfigurationsMax}')
                print(f'LMCT changed from False to True: nConfigurations changed from 1 to 2')
            elif name == '3d-Ligands Hybridization (MLCT)':
                self.parent.q.nConfigurationsMax = self.parent.nElectrons + 1
                self.parent.q.nConfigurations = 2
                print(f'MLCT changed from False to True: nConfigurationsMax changed from 1 to {self.parent.nConfigurationsMax}')
                print(f'MLCT changed from False to True: nConfigurations changed from 1 to 2')

        elif self.store[name] == True and value == False:  # from True to False
            # set value
            self.store[name] = False
            
            # Determine the maximum number of allowed configurations.
            if name == '3d-Ligands Hybridization (LMCT)':
                print(f'LMCT changed from True to False: nConfigurationsMax changed from {self.parent.nConfigurationsMax} to 1')
                print(f'LMCT changed from True to False: nConfigurations changed from {self.parent.nConfigurations} to 1')
                self.parent.q.nConfigurationsMax = 1
                self.parent.q.nConfigurations    = 1
            elif name == '3d-Ligands Hybridization (MLCT)':
                self.parent.q.nConfigurationsMax = 1
                self.parent.q.nConfigurations    = 1
                print(f'MLCT changed from True to False: nConfigurationsMax changed from {self.parent.nConfigurationsMax} to 1')
                print(f'MLCT changed from True to False: nConfigurations changed from {self.parent.nConfigurations} to 1')
        else:
            pass


    def __delitem__(self, key):
        raise AttributeError('Items cannot be deleted')

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len({name:self.store[name] for name in self.store if self.store[name] is not None})

    # def check_hybridization(self):
    #     lmct = False
    #     mlct = False
    #     for item in self.store:
    #         if item.endswith('(LMCT)'):
    #             lmct = self.store[item]
    #         elif item.endswith('(MLCT)'):
    #             mlct = self.store[item]
    #     if lmct and mlct:
    #         for item in self.store:
    #             if item.endswith('(LMCT)') or item.endswith('(MLCT)'):
    #                 self.store[item] = False
    #         raise ValueError('Ligands Hybridization LMCT and MLCT cannot be both True.\nSwitching both to False.')

    def export(self):
        temp = {}
        for item in self.store:
            if self.store[item]:
                temp[item] = 2
            else:
                temp[item] = 0
        return temp

class _hamiltonianData(MutableMapping):
    def __init__(self, initial):
        self.store = dict(copy.deepcopy(initial))
        for key in self.store:
            if type(self.store[key]) != float and type(self.store[key]) != list:
                self.store[key] = _hamiltonianData(self.store[key])
        self.initial = dict(copy.deepcopy(initial))

        self.sync = False

    def __str__(self):
        # toprint = str(self.store).replace('\':', '\':\n'+' '*10).replace(', \'', ', \n'+' '*0+'\'')
        # toprint = toprint.replace('(\'Final', '\n'+' '*11+'\'Final').replace('), (\'', ',\n'+' '*20+'\'')
        # toprint = toprint.replace('odict([(', '').replace('Hamiltonian\', \'', 'Hamiltonian\',\n'+' '*20+'\'')
        # toprint = toprint.replace(')]))])', '')
        # toprint = toprint.replace(')]))', '')
        # return toprint

        i = 8
        toprint = str(self.store).replace('\': {\'Initial', '\':\n'+' '*i+'{\'Initial')  # initial
        toprint = toprint.replace('}, \'Intermediate', '\':\n'+' '*i+'\'Intermediate')  # Intermediate
        toprint = toprint.replace('}, \'Final', '},\n'+' '*(i+1)+'\'Final')  # final
        toprint = toprint.replace('}}, \'', '\n \'')  # terms
        toprint = toprint.replace(', \'', '\n'+' '*i*2 +'\'')  # values
        toprint = toprint.replace('\': {\'', '\':\n'+' '*i*2 +'\'')  # values from the first line
        return toprint

    def __repr__(self):
        return str(self.store)

    def __getitem__(self, name):
        return self.store[name]

    def __setitem__(self, name, value):
        if name not in self:
            raise ValueError(f'New terms cannot be created.\nInvalid term: {name}\nValid terms are: {list(self.keys())}')
        if type(self.store[name]) != type(value):
            if (type(self.store[name]) == float and type(value) == int) or (type(self.store[name]) == int and type(value) == float):
                pass
            else:
                raise ValueError(f'New value does not have a valid type.\nOld value: {self.store[name]}\nOld type: {type(self.store[name])}\nNew value: {value}\nNew type: {type(value)}')
        self.store[name] = value

    def __delitem__(self, key):
        raise AttributeError('Itens cannot be deleted')

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len({name:self.store[name] for name in self.store if self.store[name] is not None})

    def reset(self):
        self.store = copy.deepcopy(self.initial)


# %% Main class ================================================================
class Calculation():
    """Initialize a Crispy/Quanty calculation object.

    Args:
        element (string, optional): transition metals, lanthanides and actinides.
            default is 'Ni'.
        charge (string, optional): suitable oxidation state of the element as
            '1+', '2+' and so on. default depends on the element.
        symmetry (string, optional): local symmetry. Possible values are 'Oh',
            'D4h', 'Td', 'C3v', and 'D3h'. default depends on the element.
        experiment (string, optional): experiment to simulate spectrum. Possible
            values are 'XAS', 'XES', 'XPS', and 'RIXS'. default is 'XAS'.
        edge (string, optional): investigated edge.
            * For 'XAS' and 'XPS', possible values are 'K (1s)', 'L1 (2s)', 'L2,3 (2p)', 'M1 (3s)', and 'M2,3 (3p)'.
            * For 'XES', possible values are 'Ka (1s2p)' and 'Kb (1s3p)'.
            * For 'RIXS', possible values are 'K-L2,3 (1s2p)', 'K-M2,3 (1s3p)', and 'L2,3-M4,5 (2p3d)'
            default is depends on the experiment type and element.

        toCalculate (list, optional): type of spectrum to calculate. default is ['Isotropic']
            * For 'XAS', possible values are 'Circular Dichroism' (cir, cd), 
            'Isotropic' (i, iso), 'Linear Dichroism' (lin, ld).
            * For 'XES' and 'XPS', possible value is 'Isotropic'.
            * For 'RIXS', possible values are 'Isotropic', 'Linear Dichroism'.
            Note: 'Linear Dichroism' for RIXS is not a crispy feature and is 
            implemented by this object.
        nPsis (number, optional): number of states to calculate. If None, 
            nPsiAuto will be set to 1 and a suitable value will be calculated. 
            Default is None.
        nPsisAuto (int, optional): If 1, a suitable value will be picked for 
            nPsis. Default is 1.

        filepath_lua (str or pathlib.Path, optional): filepath to save .lua file. 
            If extension .lua is not present, extension .lua is added.
            default depends on element, charge, etc... example: Cr3+_Oh_2p3d_RIXS.lua
        filepath_spec (str or pathlib.Path, optional): filepath to save spectrum.
            This filepath will be written in the .lua file. 
            If extension .spec is not present, extension .spec is added.
            default depends on element, charge, etc... 
            example: Cr3+_Oh_2p3d_RIXS_iso.spec
        filepath_par (str or pathlib.Path, optional): filepath to save parameter
            file (parameter list file).
            If extension .par is not present, extension .par is added.
            default depends on element, charge, etc... 
            example: Cr3+_Oh_2p3d_RIXS_iso.spec

        magneticField (number, optional): Magnetic field value in Tesla. If zero
            or None, a very small magnetic field (0.002 T) will be used to
            have nice expected values for observables. To force a zero magnetic
            field you have to turn it off via the ``hamiltonianState``. default
            is 0.002 T. The direction of the magnetic quantization axis is 
            defined by k1.
        temperature (number, optional): temperature in Kelvin. default is 10 K.
            If temperature is zero, nPsi is set to 1 and nPsiAuto to 0.

        xLorentzian (number, tuple, or list, optional):
            * For 'XAS', it is the spectral broadening for the first and second
                half of the spectrum. default depends on the element, oxidation
                state, etc. WARNING: as for now, the second number in the xLorentzian
                applies to both edges.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', n.a.
        yLorentzian (tuple or list, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', n.a. (check)
            * For 'RIXS', the first value is the broadening of the energy transfer
                axis of the RIXS spectrum. The second value is the broadening of
                the incident photon energy axis (equivalent to the XAS broadening).
                default depends on the element, oxidation state, etc. The second
                value is not implemented in the crispy interface.

        k1 (tuple, optional): magnetic quantization axis. default is (0, 0, 1).
        eps11 (tuple, optional): vertical incoming photon polarization vector. 
            default is (0, 1, 0).
        eps12 (tuple, optional): horizontal incoming photon polarization vector. 
            It should be perpendicular to eps11. default is (1, 0, 0)
        
        circular polarization vectors are calculated based on eps11 and eps12:
            er = {sqrt(1/2) * (eps12[1] - I * eps11[1]),
                  sqrt(1/2) * (eps12[2] - I * eps11[2]),
                  sqrt(1/2) * (eps12[3] - I * eps11[3])}

            el = {-sqrt(1/2) * (eps12[1] + I * eps11[1]),
                  -sqrt(1/2) * (eps12[2] + I * eps11[2]),
                  -sqrt(1/2) * (eps12[3] + I * eps11[3])}
       
        k2 (tuple, optional): scattered wavevector. default is (0, 0, 1). 
            I'm not sure if this is used ever (check)
        eps21 (tuple, optional): Only for RIXS. Sigma scattered photon polarization vector. 
            default is (0, 1, 0).
        eps22 (tuple, optional): Only for RIXS. Pi scattered photon polarization vector.
            It should be perpendicular to eps21. default is (1, 0, 0)

        xMin (number, optional):
            * For 'XAS', incident photon energy minimum value (in eV).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', incident photon energy minimum value (in eV).
            If None, a suitable value depending on the element will
            be chosen. default is None.
        xMax (number, optional):
            * For 'XAS', incident photon energy maximum value (in eV).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', incident photon energy maximum value (in eV).
            If None, a suitable value depending on the element will
            be chosen. default is None.
        xNPoints (number, optional):
            * For 'XAS', number of incident photon energy between xMin and 
            Xmax (number of data points in the XAS spectrum).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', number of incident photon energies between xMin and 
            Xmax.
            If None, a suitable value depending on the element will
            be chosen. default is None.
        yMin (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', energy transfer minimum value (in eV).
            If None, a suitable value depending on the element will
            be chosen. default is None.
        yMax (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', energy transfer maximum value (in eV).
            If None, a suitable value depending on the element will
            be chosen. default is None.
        yNPoints (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', number of energy transfer data points.
            If None, a suitable value depending on the element will
            be chosen. default is None.

        Some extra arguments that for enhanced funcionality (ONLY FOR Cu 2+):

        print_initial_wavefunctions (bool, optional): If True, will print a 
        comprehensive table with determinants, spherical
         harmonics, and factors for the initial calculated wavefunctions
         (eigenvectors of the initial hamiltonian). This is only implemented for 3d metals 
         and will raise an error otherwise. Default is False.
        filepath_for_saving_initial_wavefunctions (string, None, or Path, optional): will 
         save to a file the comprehensive table with determinants, spherical
         harmonics, and factors for the initial calculated wavefunctions
         (eigenvectors of the initial hamiltonian). This is only implemented for 3d metals 
         and will raise an error otherwise. Default is None.
        filepath_for_saving_all_eigenstates_of_initial_hamiltonian (string, None, or Path, optional):
        Same as filepath_for_saving_initial_wavefunctions, but will save all 
        eigenfunctions of the initial hamiltonian.

    Attributes:
        All initial args are also attributes, plus we have extra attrs to be 
        edited after creating a Calculation object:

        nConfigurations (int, optional): number of configurations. If None,
            a suitable value will be calculated. default is None. If LMCT or
            MLCT are False in hamiltonianState, then the nConfigurations is 
            always 1 (there is only one final configuration). If LMCT or MLCT 
            are True, then there could be multiple final states where the ligand
            lends the Metal a electron (or more) and vice-versa. The maximum number
            of final states depends on how many holes the Metal or the ligand has.
            To check the maximum number of final states allowed, check q.nConfigurationsMax
        nConfigurationsMax (int, read only): Maximum number of final configuration allowed.
            See q.nConfigurations.
        hamiltonianState (dict, optional): a dictionary that turns on or off the
            different contributions to the hamiltonian. default is such that
            'Atomic', 'Crystal Field', and 'Magnetic Field' terms will be True.
        hamiltonianData (dict, optional): dictionary with values for the strength
            of each different contributions to the hamiltonian. The default value
            ise different for each element and charge stat. Values are in eV.

    Methods:
        save_input(): Create input file for Quanty.
        run(): Create quanty calculation.

        load_spectrum(): Load calculated spectrum from file.

            if Linear Dichroism XAS, load [spectrum for eps11 (v), spectrum for eps12 (h), subtracted spectrum (v-h)]
            if Circular Dichroism XAS, load [spectrum for left (l), spectrum for right (r), subtracted spectrum (l-r)]
            if Isotropic XAS, load isotropic spectrum
            ...


        get_parameters(): Returns a dictionary with all calculation parameters.
        save_parameters(): Save calculation parameters to a file.
        load_parameters(): Load calculation parameters from file.
    """
    def __init__(self, element      = 'Ni',
                         charge     = '2+',
                         symmetry   = None,
                         experiment = None,
                         edge       = None,
                         #
                         toCalculate     = 'Isotropic',
                         nPsis           = None,
                         nPsisAuto       = 1,
                        #  nConfigurations = 1,
                         filepath_lua    = None,
                         filepath_spec   = None,
                         filepath_par    = None,
                         #
                         magneticField = 0,
                         temperature   = 10,
                         #
                         xLorentzian = None,
                         yLorentzian = None,
                         #
                         k1    = (0, 0, 1),
                         eps11 = (0, 1, 0),
                         eps12 = (1, 0, 0),
                         k2    = (0, 0, 1),
                         eps21 = (0, 1, 0),
                         eps22 = (1, 0, 0),
                         #
                         xMin     = None,
                         xMax     = None,
                         xNPoints = None,
                         yMin     = None,
                         yMax     = None,
                         yNPoints = None,
                         ):
        # crispy attributes
        self.verbosity   = '0x0000'
        self.denseBorder = '2000'

        # primary attributes
        self._set_primary_attributes(element, charge, symmetry, experiment, edge)  # sets up self.q

        # hamiltonian
        self.hamiltonianState = _hamiltonianState(self.q.hamiltonianState, parent=self)
        self.hamiltonianData  = _hamiltonianData(self.q.hamiltonianData)

        # calculation attributes
        self.toCalculate     = toCalculate
        self.nPsis           = nPsis
        self.nPsisAuto       = nPsisAuto
        self.nConfigurations = 1  # this is always 1 in Crispy if LMCT or MLCT is False
        self.filepath_lua    = filepath_lua
        self.filepath_spec   = filepath_spec
        self.filepath_par    = filepath_par

        # experiment attributes
        self.temperature   = temperature
        self.magneticField = magneticField 

        # spectrum attributes
        self.xMin     = xMin
        self.xMax     = xMax
        self.xNPoints = xNPoints
        self.yMin     = yMin
        self.yMax     = yMax
        self.yNPoints = yNPoints

        # orientation attributes
        self.k1    = k1
        self.eps11 = eps11
        self.eps12 = eps12

        self.k2    = k2
        self.eps21 = eps21
        self.eps22 = eps22

        # broadening attributes
        self.xLorentzian = xLorentzian
        self.yLorentzian = yLorentzian

    def _set_primary_attributes(self, element, charge, symmetry, experiment, edge):
        """Validate and set primary attributes."""

        # validation
        if element is None:
            element = 'Ni'
        assert element in settings.default, f'Not a valid element.\nValid elements are: {tuple(settings.default.keys())}'

        valid_charges = tuple(settings.default[element]['charges'].keys())
        if charge is None:
            charge = valid_charges[0]
        assert charge in valid_charges, f'Not a valid oxidation state for {element}\nValid oxidation states are: {valid_charges}'

        valid_symmetries = tuple(settings.default[element]['charges'][charge]['symmetries'].keys())
        if symmetry is None:
            symmetry = valid_symmetries[0]
        assert symmetry in valid_symmetries, f'Not a valid symmetry for {element} {charge}\nValid symmetries are: {valid_symmetries}'

        valid_experiments = tuple(settings.default[element]['charges'][charge]['symmetries'][symmetry]['experiments'].keys())
        if experiment is None:
            experiment = valid_experiments[0]
        assert experiment in valid_experiments, f'Not a valid experiment for {element} {charge} {symmetry}\nValid experiments are: {valid_experiments}'

        valid_edges = tuple(settings.default[element]['charges'][charge]['symmetries'][symmetry]['experiments'][experiment]['edges'].keys())
        if edge is None:
            edge = valid_edges[0]
        assert edge in valid_edges, f'Not a valid edge for {element} {charge} {symmetry} {experiment}\nValid edges are: {valid_edges}'

        # initialize calculation object
        self.q = settings.QuantyCalculation(element=element,
                                            charge=charge,
                                            symmetry=symmetry,
                                            experiment=experiment,
                                            edge=edge)
        # set verbosity (this is set the same as in crispy)
        self.q.verbosity =  self.verbosity

        # set border (this is set the same as in crispy)
        self.q.denseBorder = self.denseBorder

    # % primary attributes
    @property
    def element(self):
        return self.q.element
    @element.setter
    def element(self, value):
        raise AttributeError('Primary attributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new Calculation() object')
    @element.deleter
    def element(self):
        raise AttributeError('Cannot delete object.')

    @property
    def charge(self):
        return self.q.charge
    @charge.setter
    def charge(self, value):
        raise AttributeError('Primary attributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new Calculation() object')
    @charge.deleter
    def charge(self):
        raise AttributeError('Cannot delete object.')

    @property
    def symmetry(self):
        return self.q.symmetry
    @symmetry.setter
    def symmetry(self, value):
        raise AttributeError('Primary attributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new Calculation() object')
    @symmetry.deleter
    def symmetry(self):
        raise AttributeError('Cannot delete object.')

    @property
    def experiment(self):
        return self.q.experiment
    @experiment.setter
    def experiment(self, value):
        raise AttributeError('Primary attributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new Calculation() object')
    @experiment.deleter
    def experiment(self):
        raise AttributeError('Cannot delete object.')

    @property
    def edge(self):
        return self.q.edge
    @edge.setter
    def edge(self, value):
        raise AttributeError('Primary attributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new Calculation() object')
    @edge.deleter
    def edge(self):
        raise AttributeError('Cannot delete object.')
    
    @property
    def nElectrons(self):
        return self.q.nElectrons
    @nElectrons.setter
    def nElectrons(self, value):
        raise AttributeError('nElectrons is defined with primary attributes (element, charge, symmetry, experiment, edge) and cannot be changed.\nPlease, start a new Calculation() object')
    @nElectrons.deleter
    def nElectrons(self):
        raise AttributeError('Cannot delete object.')
    


    # % calculation attributes
    @property
    def toCalculate(self):
        return self.q.spectra.toCalculate[0]
    @toCalculate.setter
    def toCalculate(self, value):
        # validation
        error = "Invalid value for toCalculate\nValid values are:\nXAS: 'Circular Dichroism', 'Isotropic', 'Linear Dichroism'\nXES: 'Isotropic'\nXPS: 'Isotropic'\nRIXS: 'Isotropic', 'Linear Dichroism', 'Polarimeter'"
        assert type(value) == str, error
        assert value.lower().startswith(("cir", "cd", "lin", "ld", "iso", 'i', 'pol', 'polarimeter')), error
        if self.experiment == 'RIXS':
            assert value.lower().startswith(("lin", "ld", "iso", 'i', 'pol', 'polarimeter')), error
        if self.experiment == 'XES' or self.experiment == 'XPS':
            assert value.lower().startswith(("iso", 'i')), error

        # convert value to crispy syntax
        if value.lower().startswith(("cir", "cd")):
            value = 'Circular Dichroism'
        elif value.lower().startswith(("lin", "ld")):
            value = 'Linear Dichroism'
        elif value.lower().startswith(("i", "iso")):
            value = 'Isotropic'
        elif value.lower().startswith(("pol", "pola")):
            value = 'Polarimeter'

        # set value
        self.q.spectra.toCalculate        = [value]
        self.q.spectra.toCalculateChecked = [value]
    @toCalculate.deleter
    def toCalculate(self):
        raise AttributeError('Cannot delete object.')

    @property
    def nPsis(self):
        return self.q.nPsis
    @nPsis.setter
    def nPsis(self, value):
        if value is None:
            self.q.nPsisAuto = 1
            value = self.q.nPsisMax
        else:
            assert value > 0, 'The number of states must be larger than zero.'
            assert value <= self.q.nPsisMax, f'The selected number of states exceeds the maximum.\nMaximum {self.q.nPsisMax}'
        self.q.nPsis     = value
        self.q.nPsisAuto = 0
    @nPsis.deleter
    def nPsis(self):
        raise AttributeError('Cannot delete object.')

    @property
    def nPsisAuto(self):
        return self.q.nPsisAuto
    @nPsisAuto.setter
    def nPsisAuto(self, value):
        if value == 0:
            self.q.nPsisAuto = 0
        elif value == 1:
            self.q.nPsisAuto = 1
        else:
            raise ValueError('nPsiAuto can only be 0 or 1')
    @nPsisAuto.deleter
    def nPsisAuto(self):
        raise AttributeError('Cannot delete object.')

    @property
    def nConfigurationsMax(self):
        return self.q.nConfigurationsMax
    @nConfigurationsMax.setter
    def nConfigurationsMax(self, value):
        raise AttributeError('Cannot manually edit nConfigurationsMax object.')
    @nConfigurationsMax.deleter
    def nConfigurationsMax(self):
        raise AttributeError('Cannot delete object.')

    @property
    def nConfigurations(self):
        return self.q.nConfigurations
    @nConfigurations.setter
    def nConfigurations(self, value):
        assert value <= self.nConfigurationsMax, f'The maximum number of configurations is {self.nConfigurationsMax}'
        assert value >= 1, f'The minimum number of configurations is 1'
        assert isinstance(value, int), f'nConfigurations must be a int, not type: {type(value)}'
        self.q.nConfigurations = value
    @nConfigurations.deleter
    def nConfigurations(self):
        raise AttributeError('Cannot delete object.')
    
    



    @property
    def filepath_lua(self):
        return self._filepath_lua
    @filepath_lua.setter
    def filepath_lua(self, value):
        if value is None:
            value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['template name']
            value = Path(value).with_suffix('.lua')
        assert type(value) == str or isinstance(value, Path), 'filepath_lua must be a string or pathlib.Path.'
        self._filepath_lua = value
    @filepath_lua.deleter
    def filepath_lua(self):
        raise AttributeError('Cannot delete object.')

    @property
    def filepath_spec(self):
        return self._filepath_spec
    @filepath_spec.setter
    def filepath_spec(self, value):
        if value is None:
            value = Path(self.filepath_lua).with_suffix('.spec')
        assert type(value) == str or isinstance(value, Path), 'filepath_lua must be a string or pathlib.Path.'
        self._filepath_spec = value
    @filepath_spec.deleter
    def filepath_spec(self):
        raise AttributeError('Cannot delete object.')

    @property
    def filepath_par(self):
        return self._filepath_par
    @filepath_par.setter
    def filepath_par(self, value):
        if value is None:
            value = Path(self.filepath_lua).with_suffix('.par')
        assert type(value) == str or isinstance(value, Path), 'filepath_par must be a string or pathlib.Path.'
        self._filepath_par = value
    @filepath_par.deleter
    def filepath_par(self):
        raise AttributeError('Cannot delete object.')

    # % spectrum attributes
    @property
    def xMin(self):
        return self.q.xMin
    @xMin.setter
    def xMin(self, value):
        if value is None:
            value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][1]
        assert value > 0, 'xMin cannot be negative'
        self.q.xMin = value
    @xMin.deleter
    def xMin(self):
        raise AttributeError('Cannot delete object.')

    @property
    def xMax(self):
        return self.q.xMax
    @xMax.setter
    def xMax(self, value):
        if value is None:
            value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][2]
        assert value > 0, 'xMax cannot be negative'
        self.q.xMax = value
    @xMax.deleter
    def xMax(self):
        raise AttributeError('Cannot delete object.')

    @property
    def xNPoints(self):
        return self.q.xNPoints + 1
    @xNPoints.setter
    def xNPoints(self, value):
        if value is None:
            value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][3]
        assert value >= 2, 'xNPoints cannot be less than 2.\nThe CreateResonantSpectra() function from Quanty prevents xNPoints to be less than 2.'

        value = int(value - 1)

        self.q.xNPoints = value
    @xNPoints.deleter
    def xNPoints(self):
        raise AttributeError('Cannot delete object.')

    @property
    def yMin(self):
        return self.q.yMin
    @yMin.setter
    def yMin(self, value):
        if value is None:
            if self.experiment != 'RIXS':
                value = None
            else:
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][1][1]
        self.q.yMin = value
    @yMin.deleter
    def yMin(self):
        raise AttributeError('Cannot delete object.')

    @property
    def yMax(self):
        return self.q.yMax
    @yMax.setter
    def yMax(self, value):
        if value is None:
            if self.experiment != 'RIXS':
                value = None
            else:
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][1][2]
        self.q.yMax = value
    @yMax.deleter
    def yMax(self):
        raise AttributeError('Cannot delete object.')

    @property
    def yNPoints(self):
        try:
            return self.q.yNPoints + 1
        except TypeError:
            return self.q.yNPoints
    @yNPoints.setter
    def yNPoints(self, value):
        if value is None:
            if self.experiment != 'RIXS':
                value = None
                self.q.yNPoints = None
                return
            else:
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][1][3]
        assert value >= 2, 'yNPoints cannot be less than 2.\nThe CreateResonantSpectra() function from Quanty prevents yNPoints to be less than 2.'
        
        value = int(value - 1)

        self.q.yNPoints = value
    @yNPoints.deleter
    def yNPoints(self):
        raise AttributeError('Cannot delete object.')

    # % broadening attributes
    @property
    def xLorentzian(self):
        return self.q.xLorentzian
    @xLorentzian.setter
    def xLorentzian(self, value):
        if value is None:
            if self.experiment == 'RIXS':
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][5]
            else:
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][5]            
        try:
            if value < 0:
                raise ValueError('Broadening cannot be negative.')
            value = (value, value)
        except TypeError:
            if any([v < 0 for v in value]):
                raise ValueError('Broadening cannot be negative.')
        self.q.xLorentzian = value
    @xLorentzian.deleter
    def xLorentzian(self):
        raise AttributeError('Cannot delete object.')

    @property
    def yLorentzian(self):
        return self.q.yLorentzian
    @yLorentzian.setter
    def yLorentzian(self, value):
        if value is None:
            if self.experiment == 'RIXS':
                value = settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][1][5]
            else:
                value = (0, 0)            
        try:
            if value < 0:
                raise ValueError('Broadening cannot be negative.')
            value = (value, settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][2][5][1])
        except TypeError:
            if any([v < 0 for v in value]):
                raise ValueError('Broadening cannot be negative.')
        self.q.yLorentzian = value
    @yLorentzian.deleter
    def yLorentzian(self):
        raise AttributeError('Cannot delete object.')

    # % experiment attributes
    @property
    def temperature(self):
        return self.q.temperature
    @temperature.setter
    def temperature(self, value):
        assert value >= 0, 'Temperature cannot be negative.'
        if value == 0:
            print('Temperature = 0\nnPsi set to 1.')
            self.nPsis = 1
            self.nPsisAuto = 0
        self.q.temperature = value
    @temperature.deleter
    def temperature(self):
        raise AttributeError('Cannot delete object.')

    @property
    def magneticField(self):
        return self.q.magneticField
    @magneticField.setter
    def magneticField(self, value):
        # print(value)
        # small = np.finfo(np.float32).eps  # ~1.19e-7 
        if value is None or value == 0:
            value = 0.002
        elif value < 0:
            raise ValueError('Magnetic field value must be positive.')
        elif value < 0.002:
            raise ValueError('Magnetic field cannot be smaller than 0.002 T.\nTurn off the magnetic field hamiltonian using hamiltonianState["Magnetic Field"]')     
       
        self.q.magneticField = value
        self._update_magnetic_field_hamiltonian_data()      
    @magneticField.deleter
    def magneticField(self):
        raise AttributeError('Cannot delete object.')

    # % orientation attributes
    @property
    def k1(self):
        return self.q.k1
    @k1.setter
    def k1(self, value):
        assert len(value) == 3, 'k1 must be a vector, like [0, 1, 0].'
        self.q.k1 = _normalize(value)
        self._update_magnetic_field_hamiltonian_data()
    @k1.deleter
    def k1(self):
        raise AttributeError('Cannot delete object.')

    @property
    def eps11(self):
        return self.q.eps11
    @eps11.setter
    def eps11(self, value):
        assert len(value) == 3, 'eps11 must be a vector, like [0, 1, 0].'
        self.q.eps11 = _normalize(value)
    @eps11.deleter
    def eps11(self):
        raise AttributeError('Cannot delete object.')

    @property
    def eps12(self):
        return self.q.eps12
    @eps12.setter
    def eps12(self, value):
        assert len(value) == 3, 'eps12 must be a vector, like [0, 1, 0].'
        self.q.eps12 = _normalize(value)
        # raise ValueError('Cannot edit eps12.\nIts value is perpendicular to k1 and eps11.')
    @eps12.deleter
    def eps12(self):
        raise AttributeError('Cannot delete object.')

    @property
    def k2(self):
        return self.q.k2
    @k2.setter
    def k2(self, value):
        assert len(value) == 3, 'k2 must be a vector, like [0, 1, 0].'
        self.q.k2 = _normalize(value)
    @k2.deleter
    def k2(self):
        raise AttributeError('Cannot delete object.')

    @property
    def eps21(self):
        return self.q.eps21
    @eps21.setter
    def eps21(self, value):
        assert len(value) == 3, 'eps21 must be a vector, like [0, 1, 0].'
        self.q.eps21 = _normalize(value)
    @eps21.deleter
    def eps21(self):
        raise AttributeError('Cannot delete object.')

    @property
    def eps22(self):
        return self.q.eps22
    @eps22.setter
    def eps22(self, value):
        assert len(value) == 3, 'eps22 must be a vector, like [0, 1, 0].'
        self.q.eps22 = _normalize(value)
        # raise ValueError('Cannot edit eps22.\nIts value is perpendicular to k1 and eps11.')
    @eps22.deleter
    def eps22(self):
        raise AttributeError('Cannot delete object.')

    # % extra attr
    @property
    def configurations(self):
        return settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['configurations']
    @configurations.setter
    def configurations(self, value):
        raise AttributeError('Attribute is read-only.')
    @configurations.deleter
    def configurations(self):
        raise AttributeError('Cannot delete object.')
    
    @property
    def resonance(self):
        return settings.default[self.element]['charges'][self.charge]['symmetries'][self.symmetry]['experiments'][self.experiment]['edges'][self.edge]['axes'][0][4]
    @resonance.setter
    def resonance(self, value):
        raise AttributeError('Attribute is read-only.')
    @resonance.deleter
    def resonance(self):
        raise AttributeError('Cannot delete object.')
    

    # % support
    def _update_magnetic_field_hamiltonian_data(self):
        value = self.magneticField
        k1    = np.array(self.k1)

        TESLA_TO_EV = 5.788e-05
        value = value * TESLA_TO_EV
        
        # updating hamiltonian data
        configurations = self.hamiltonianData['Magnetic Field']
        for configuration in configurations:
            parameters = self.hamiltonianData['Magnetic Field'][configuration]
            for i, parameter in enumerate(parameters):
                value2 = float(value * np.abs(k1[i]))
                # value2 = float(value * k1[i] * TESLA_TO_EV)

                # fix negative zero problem (same as in crispy)
                if abs(value2) == 0.0:
                    value2 = 0.0
                
                self.hamiltonianData['Magnetic Field'][configuration][parameter] = value2
   
    def _calculate_incident_energies(self):
        """For RIXS, calculate incident energies"""
        return np.linspace(self.xMin, self.xMax, self.xNPoints)
    
    # % Core methods
    def save_input(self, print_initial_wavefunctions=False, 
                   filepath_for_saving_initial_wavefunctions=None,
                   filepath_for_saving_all_eigenstates_of_initial_hamiltonian=None):
        """Create and save a Quanty input file (.lua).

        Example:
            q = sg4.Calculation(element='Cu', charge='2+', symmetry='D4h', experiment='RIXS', edge='L2,3-M4,5 (2p3d)')
            q.filepath_lua = 'Document/test_filepath'
            q.save_input()

        Args:
            print_initial_wavefunctions (bool, optional): If True, the initial 
                wavefunction will be printed. Default is False. AS FOR RIGHT NOW,
                THIS ONLY WORKS FOR Cu2+. 
            filepath_for_saving_initial_wavefunctions (None, str, or path, optional): 
                If str or Path, a file will be saved with the initial wavefunction.
                Default is None. AS FOR RIGHT NOW, THIS ONLY WORKS FOR Cu2+. 
            filepath_for_saving_all_eigenstates_of_initial_hamiltonian (None, str, or path, optional): 
                If str or Path, a file will be saved with the eigenstate of the initial hamiltonian.
                Default is None. AS FOR RIGHT NOW, THIS ONLY WORKS FOR Cu2+. 

        Warning:
            print_initial_wavefunctions, filepath_for_saving_initial_wavefunctions, and
            filepath_for_saving_all_eigenstates_of_initial_hamiltonian only works
            for Cu2+. Implementation for other ions should be easy, but I didn't 
            get to do it yet.

        Returns:
            None
        """
        # hamiltonian ==================================
        self.q.hamiltonianState = {key:(1 if value else 0) for key, value in self.hamiltonianState.items()}
        self.q.hamiltonianData  = copy.deepcopy(self.hamiltonianData)

        # transfer filepath_lua to basename =============
        filepath_lua = Path(self.filepath_lua).with_suffix('')

        if is_windows:
            parts = list(filepath_lua.parts)
            for i, part in enumerate(parts):
                if '\\' in part:
                    parts[i] = part.replace('\\', '')
            self.q.baseName = str(r'\\'.join(parts))
        else:
            self.q.baseName = str(filepath_lua)

        # Temporarily change toCalculate if Polarimeter
        polarimeter = False
        if self.toCalculate == 'Polarimeter':
            polarimeter = True
            self.toCalculate = 'Isotropic'

        # save input ====================================
        self.q.saveInput()   

        # change polarimeter back
        if polarimeter:
            self.toCalculate = 'Polarimeter'

        # replacements ==================================
        filepath_lua  = filepath_lua.with_suffix('.lua')
        filepath_spec = Path(self.filepath_spec).with_suffix('')

        # filepath spectrum
        if is_windows:
            parts = list(filepath_spec.parts)
            for i, part in enumerate(parts):
                if '\\' in part:
                    parts[i] = part.replace('\\', '')
            filepath_spec = str(r'\\'.join(parts))
        else:
            filepath_spec = str(filepath_spec)

        if self.experiment == 'XPS' or self.experiment == 'XES':
            pattern = r"Giso.Print({{'file', '" + self.q.baseName + r"_iso.spec'}})"
            subst   = r"Giso.Print({{'file', '" + filepath_spec   + r"_iso.spec'}})"
            replace(filepath_lua, pattern, subst)
        elif self.experiment == 'XAS':
            pattern = r"G.Print({{'file', '" + self.q.baseName + r"_' .. suffix .. '.spec'}})"
            subst   = r"G.Print({{'file', '" + filepath_spec   + r"_' .. suffix .. '.spec'}})"
            replace(filepath_lua, pattern, subst)

        elif self.experiment == 'RIXS' and self.toCalculate == 'Isotropic':
            pattern = r"Giso.Print({{'file', '" + self.q.baseName + r"_iso.spec'}})"
            subst   = r"Giso.Print({{'file', '" + filepath_spec   + r"_iso.spec'}})"
            replace(filepath_lua, pattern, subst)
        elif self.experiment == 'RIXS' and (self.toCalculate == 'Linear Dichroism' or self.toCalculate == 'Polarimeter'):
            pattern = r"Giso.Print({{'file', '" + self.q.baseName + r"_iso.spec'}})"
            subst   = r"Giso.Print({{'file', '" + filepath_spec   + r".spec'}})"
            replace(filepath_lua, pattern, subst)

        # XAS energy shift
        if self.experiment == 'XAS':
            pattern = "SaveSpectrum(Giso, 'iso')"
            subst = "Giso.Shift(Eedge1-DeltaE)\n" + \
                    "    SaveSpectrum(Giso, 'iso')"
            replace(filepath_lua, pattern, subst)

            pattern = "SaveSpectrum(Gr, 'r')"
            subst = "Gr.Shift(Eedge1-DeltaE)\n" + \
                    "    Gl.Shift(Eedge1-DeltaE)\n" + \
                    "    SaveSpectrum(Gr, 'r')"
            replace(filepath_lua, pattern, subst)
        
            pattern = "SaveSpectrum(Gv, 'v')"
            subst = "Gv.Shift(Eedge1-DeltaE)\n" + \
                    "    Gh.Shift(Eedge1-DeltaE)\n" + \
                    "    SaveSpectrum(Gv, 'v')"
            replace(filepath_lua, pattern, subst)

        
        # LMCT fix
        if "3d-Ligands Hybridization (MLCT)" in self.hamiltonianState:
            if self.hamiltonianState["3d-Ligands Hybridization (MLCT)"]:
                if self.hamiltonianState["3d-Ligands Hybridization (LMCT)"]:
                    pattern = "H_3d_ligands_hybridization_lmct = 1\n"
                    subst = "H_3d_ligands_hybridization_lmct = 1\n" + \
                            "H_3d_ligands_hybridization_mlct = 1\n"
                else:
                    pattern = "H_3d_ligands_hybridization_lmct = 0\n"
                    subst = "H_3d_ligands_hybridization_lmct = 0\n" + \
                            "H_3d_ligands_hybridization_mlct = 1\n"
                replace(filepath_lua, pattern, subst)

        # RIXS yLorentzian
        if self.experiment == 'RIXS':
            pattern = "Gamma1 = 0.1"
            subst = f'Gamma1 = {self.q.yLorentzian[1]}'
            replace(filepath_lua, pattern, subst)

        # RIXS linear dichroism
        if self.experiment == 'RIXS' and self.toCalculate == 'Linear Dichroism':
            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            replace(filepath_lua, pattern, subst)

            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            replace(filepath_lua, pattern, subst)

            pattern = "    for j = 1, 3 * 3 do"
            subst   = "    for j = 1, 3 do"
            replace(filepath_lua, pattern, subst)

        # RIXS polarimeter
        if self.experiment == 'RIXS' and self.toCalculate == 'Polarimeter':
            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    e2 = {" + ','.join([str(x) for x in self.eps21]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    T_3d_2p = e2[1] * Tx_3d_2p + e2[2] * Ty_3d_2p + e2[3] * Tz_3d_2p\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, T_3d_2p, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            replace(filepath_lua, pattern, subst)

            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    e2 = {" + ','.join([str(x) for x in self.eps21]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    T_3d_2p = e2[1] * Tx_3d_2p + e2[2] * Ty_3d_2p + e2[3] * Tz_3d_2p\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, T_3d_2p, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            replace(filepath_lua, pattern, subst)

            pattern = "    for j = 1, 3 * 3 do"
            subst   = "    for j = 1, 1 do"
            replace(filepath_lua, pattern, subst)


        # print initial wavefunctions (ONLY FOR CU2+)
        if print_initial_wavefunctions or filepath_for_saving_initial_wavefunctions is not None or filepath_for_saving_all_eigenstates_of_initial_hamiltonian is not None:
            assert self.element == 'Cu' and self.charge == '2+', f'Sorry, as for right now, `priting/saving wavefunctions` is only available for Cu2+'

            pattern = "Verbosity(0x0000)"
            subst   = "Verbosity(0x0000)\n\n" + text_defining_functions_for_print_wavefunction_for_d4h_3d_ions
            replace(filepath_lua, pattern, subst)

            pattern = "-- Define the atomic term."
            subst   = text_defining_base_for_printing_wavefunction_for_d4h_3d_ions + '\n\n--------------------------------------------------------------------------------\n-- Define the atomic term.'
            replace(filepath_lua, pattern, subst)

        if print_initial_wavefunctions or filepath_for_saving_initial_wavefunctions is not None:
            assert self.element == 'Cu' and self.charge == '2+', f'Sorry, as for right now, `priting/saving wavefunctions` is only available for Cu2+'

            if print_initial_wavefunctions: 
                _verbose = 1
            else:
                _verbose = 0
            if filepath_for_saving_initial_wavefunctions is not None:
                _filepath = Path(filepath_for_saving_initial_wavefunctions)
                # assert _filepath.is_file(), 'filepath_for_saving_initial_wavefunctions must point to a file'

                if is_windows:
                    parts = list(_filepath.parts)
                    for i, part in enumerate(parts):
                        if '\\' in part:
                            parts[i] = part.replace('\\', '')
                    _filepath = "'" + str(r'\\'.join(parts)) + "'"
                else:
                    _filepath = "'" + str(_filepath) + "'"
            else:
                _filepath = 'dummy'
            
            pattern = "-- Normalize dZ to unity."
            subst   = f"-- print/save initial wavefunctions\npsi_i_str = print_psi(Psis_i, base_psis, base_dets, base_sphe, H_i, " + _filepath + f", {_verbose})" +  "\n\n-- Normalize dZ to unity."
            replace(filepath_lua, pattern, subst)


        if filepath_for_saving_all_eigenstates_of_initial_hamiltonian is not None:
            assert self.element == 'Cu' and self.charge == '2+', f'Sorry, as for right now, `priting/saving wavefunctions` is only available for Cu2+'
            
            if filepath_for_saving_all_eigenstates_of_initial_hamiltonian is not None:
                _filepath = Path(filepath_for_saving_all_eigenstates_of_initial_hamiltonian)
                # assert _filepath.is_file(), 'filepath_for_saving_all_eigenstates_of_initial_hamiltonian must point to a file'

                if is_windows:
                    parts = list(_filepath.parts)
                    for i, part in enumerate(parts):
                        if '\\' in part:
                            parts[i] = part.replace('\\', '')
                    _filepath = "'" + str(r'\\'.join(parts)) + "'"
                else:
                    _filepath = "'" + str(_filepath) + "'"
            else:
                _filepath = 'dummy'
            
            pattern = "dZ = {}"
            subst   = "dZ = {}\n\n-- save all eigenfunctions of H_i\n"
            subst  += "if CalculationRestrictions == nil then\n"
            subst  += "    Psis_all = Eigensystem(H_i, InitialRestrictions, NPsis)\n"
            subst  += "else\n"
            subst  += "    Psis_all = Eigensystem(H_i, InitialRestrictions, NPsis, {{'restrictions', CalculationRestrictions}})\n"
            subst  += "end\n"
            subst  += "psi_str = print_psi(Psis_all, base_psis, base_dets, base_sphe, H_i, " + _filepath + f", 0)\n"
            subst  += "------------------------------------\n"
            replace(filepath_lua, pattern, subst)

    def run(self):
        """Run Quanty.

        Returns:
            String with calculation output (stdout).
        """
        filepath = Path(self.filepath_lua).with_suffix('.lua')
        return quanty(filepath)

    # % Spectrum
    def _copy_attrs_to_spectra(self, ss):          

        if isinstance(ss, settings._br.Spectra):
            for s in ss:
                s.element    = self.element
                s.charge     = self.charge
                s.symmetry   = self.symmetry
                s.experiment = self.experiment
                s.edge       = self.edge
                s.nElectrons = self.nElectrons

                s.toCalculate        = self.toCalculate
                s.nPsis              = self.nPsis
                s.nPsisAuto          = self.nPsisAuto
                s.nConfigurations    = self.nConfigurations
                s.nConfigurationsMax = self.nConfigurationsMax
                
                s.hamiltonianData  = self.hamiltonianData
                s.hamiltonianState = self.hamiltonianState

                s.magneticField   = self.magneticField
                s.temperature     = self.temperature

                s.xLorentzian     = self.xLorentzian
                s.yLorentzian     = self.yLorentzian

                s.k1              = self.k1
                s.eps11           = self.eps11
                s.eps12           = self.eps12
                s.k2              = self.k2
                s.eps21           = self.eps21
                s.eps22           = self.eps22

                s.xMin            = self.xMin
                s.xMax            = self.xMax
                s.xNPoints        = self.xNPoints
                s.yMin            = self.yMin
                s.yMax            = self.yMax
                s.yNPoints        = self.yNPoints

            if self.experiment == 'RIXS':
                ss.E = self._calculate_incident_energies()
                for i, s in enumerate(ss):
                    s.E = ss.E[i]
        
        ss.element    = self.element
        ss.charge     = self.charge
        ss.symmetry   = self.symmetry
        ss.experiment = self.experiment
        ss.edge       = self.edge
        ss.nElectrons = self.nElectrons

        ss.toCalculate        = self.toCalculate
        ss.nPsis              = self.nPsis
        ss.nPsisAuto          = self.nPsisAuto
        ss.nConfigurations    = self.nConfigurations
        ss.nConfigurationsMax = self.nConfigurationsMax

        ss.hamiltonianData  = self.hamiltonianData
        ss.hamiltonianState = self.hamiltonianState

        ss.magneticField   = self.magneticField
        ss.temperature     = self.temperature

        ss.xLorentzian     = self.xLorentzian
        ss.yLorentzian     = self.yLorentzian

        ss.k1              = self.k1
        ss.eps11           = self.eps11
        ss.eps12           = self.eps12
        ss.k2              = self.k2
        ss.eps21           = self.eps21
        ss.eps22           = self.eps22

        ss.xMin            = self.xMin
        ss.xMax            = self.xMax
        ss.xNPoints        = self.xNPoints
        ss.yMin            = self.yMin
        ss.yMax            = self.yMax
        ss.yNPoints        = self.yNPoints

    def load_spectrum(self):
        filepath = Path(self.filepath_spec).with_suffix('.spec')

        if self.toCalculate == 'Isotropic':
            filepath = filepath.parent / str(filepath.name).replace('.spec', '_iso.spec')
            data = load_spectrum(filepath)
        elif self.toCalculate == 'Circular Dichroism':
            # filepath = filepath.parent / str(filepath.name).replace('.spec', '_r.spec')  
            # data = load_spectrum(filepath)
            # if settings.USE_BRIXS:
            #     data[0].pol = 'left'
            #     data[1].pol = 'right'
            #     data[2].pol = 'cd'
            filepaths = [filepath.parent / str(filepath.name).replace('.spec', '_l.spec'), 
                         filepath.parent / str(filepath.name).replace('.spec', '_r.spec'), 
                         filepath.parent / str(filepath.name).replace('.spec', '_cd.spec') 
                        ]  
            data = load_spectrum(*filepaths)
            if settings.USE_BRIXS:
                data[0].pol = 'left'
                data[1].pol = 'right'
                data[2].pol = 'cd'
        elif self.experiment == 'XAS' and self.toCalculate == 'Linear Dichroism':
            # filepath = filepath.parent / str(filepath.name).replace('.spec', '_v.spec')
            # data = load_spectrum(filepath)
            filepaths = [filepath.parent / str(filepath.name).replace('.spec', '_v.spec'),
                         filepath.parent / str(filepath.name).replace('.spec', '_h.spec'),
                         filepath.parent / str(filepath.name).replace('.spec', '_ld.spec')
                        ]
            data = load_spectrum(*filepaths)
            if settings.USE_BRIXS:
                data[0].pol = 'vertical'
                data[1].pol = 'horizontal'
                data[2].pol = 'ld'
        elif self.experiment == 'RIXS' and self.toCalculate == 'Linear Dichroism':
            data = load_spectrum(filepath)
        elif self.experiment == 'RIXS' and self.toCalculate == 'Polarimeter':
            data = load_spectrum(filepath)

        if settings.USE_BRIXS:
            self._copy_attrs_to_spectra(data)
            return data
        else:
            return data

    # % parameters
    def get_parameters(self):
        """Returns a dictionary with all mutable attributes."""
        p = dict(copy.deepcopy(self.hamiltonianData))
        for k1 in p:
            p[k1] = dict(p[k1])
        for k1 in p:
            for k2 in p[k1]:
                p[k1][k2] = dict(p[k1][k2])

        return dict(element             = self.element
                      ,charge           = self.charge
                      ,symmetry         = self.symmetry
                      ,experiment       = self.experiment
                      ,edge             = self.edge
                      #
                      ,toCalculate      = self.toCalculate
                      ,nPsis            = self.nPsis
                      #
                      ,magneticField    = self.magneticField
                      ,temperature      = self.temperature
                      #
                      ,xLorentzian      = self.xLorentzian
                      ,yLorentzian      = self.yLorentzian
                      #
                      ,k1               = self.k1
                      ,eps11            = self.eps11
                      ,eps12            = self.eps12
                      ,k2               = self.k2
                      ,eps21            = self.eps21
                      ,eps22            = self.eps22
                      #
                      ,xMin             = self.xMin
                      ,xMax             = self.xMax
                      ,xNPoints         = self.xNPoints
                      ,yMin             = self.yMin
                      ,yMax             = self.yMax
                      ,yNPoints         = self.yNPoints
                      #
                      ,filepath_lua     = str(self.filepath_lua)
                      ,filepath_spec    = str(self.filepath_spec)
                      ,filepath_par    = str(self.filepath_par)
                      #
                      ,hamiltonianState = dict(self.hamiltonianState)
                      ,hamiltonianData  = p
                      )

    def save_parameters(self):
        """Save calculation parameters in a text file.

        Returns:
            None

        See Also:
            :py:func:`load_parameters`
        """
        filepath = Path(self.filepath_par).with_suffix('.par')
        pretty_print = True
        par = self.get_parameters()

        # assert filepath.exists(), 'Filepath does not exist'
        # assert filepath.is_file(), 'Filepath does point to a file'

        with open(str(filepath), 'w') as file:
            if pretty_print:
                file.write(json.dumps(par, indent=4, sort_keys=False))
            else:
                file.write(json.dumps(par))

    def load_parameters(self):
        """Load calculation parameters.

        Args:
            filepath (string or pathlib.Path): filepath.

        Returns:
            None

        See Also:
            :py:func:`save_parameters`
        """
        filepath = Path(self.filepath_par).with_suffix('.par')

        with open(str(filepath), 'r') as file:
            par = json.load(file)

        hamiltonianData  = par.pop('hamiltonianData')
        hamiltonianState = par.pop('hamiltonianState')

        self.__init__(**par)
        for k1 in hamiltonianState:
            self.hamiltonianState[k1] = hamiltonianState[k1]
        for k1 in hamiltonianData:
            for k2 in hamiltonianData[k1]:
                for k3 in hamiltonianData[k1][k2]:
                    self.hamiltonianData[k1][k2][k3] = hamiltonianData[k1][k2][k3]

    # % additional Quanty code ====================================================
    def add_code_print_initial_wavefunctions(self):
        pass

    def add_code_to_save_initial_wavefunctions(self):
        pass

    def add_code_to_save_all_eigenstates_of_initial_hamiltonian(self):
        pass


        @filepath_for_saving_all_eigenstates_of_initial_hamiltonian.setter
        def filepath_for_saving_all_eigenstates_of_initial_hamiltonian(self, value):
            assert isinstance(value, str) or isinstance(value, Path) or value == None, f'filepath_for_saving_all_eigenstates_of_initial_hamiltonian needs to be None, string, or Path, not {type(value)}'
            if value is not None:
                assert self.element == 'Cu' and self.charge == '2+', f'Sorry, as for right now, `priting/saving wavefunctions` is only available for Cu2+'
                # assert '3d' == self.q.block, f'Sorry, as for right now, `saving initial wavefunctions` is only available for 3d metals, not {self.q.block}'
            self._filepath_for_saving_all_eigenstates_of_initial_hamiltonian = value
        @filepath_for_saving_all_eigenstates_of_initial_hamiltonian.deleter
        def filepath_for_saving_all_eigenstates_of_initial_hamiltonian(self):
            raise AttributeError('Cannot delete object.')

# % functions =========================================================
def replace(filepath, pattern, subst):
    """replace pattern string by another string.

    Args:
        filepath (str or path): filepath
        pattern (str): pattern to search for (must be contained in a single line)
        subst (str): string to replace pattern (can be multiline)

    Returns:
        None    
    """
    # Create temp file
    fh, abs_path = tempfile.mkstemp()
    found = False
    with os.fdopen(fh, 'w') as new_file:
        with open(filepath) as old_file:
            for line in old_file:
                if pattern in line:
                    found = True
                new_file.write(line.replace(pattern, subst))
    # Copy the file permissions from the old file to the new file
    shutil.copymode(filepath, abs_path)
    
    # Remove original file
    os.remove(filepath)
    
    # Move new file
    shutil.move(abs_path, filepath)

    if found == False:
        print(f'ERROR: Cannot find {pattern}\n\n')

def quanty(filepath):
    """Run Quanty.

    Args:
        filepath (string or pathlib.Path): path to file.

    Returns:
        Calculation output (stdout).
    """
    quanty_exe = Path(settings.QUANTY_FILEPATH)
    if is_windows:
        if quanty_exe.is_absolute():
            quanty = subprocess.Popen([f"{quanty_exe}", f"{filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            quanty = subprocess.Popen([f"./{quanty_exe} {filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif is_linux:
        if quanty_exe.is_absolute():
            quanty = subprocess.Popen([f"{quanty_exe} {filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            quanty = subprocess.Popen([f"./{quanty_exe} {filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif is_mac:
        quanty = subprocess.Popen([f"./{quanty_exe} {filepath}"], shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('here')
    print(quanty)

    
    output = quanty.stdout.read().decode("utf-8")
    print(output)

    error  = quanty.stderr.read().decode("utf-8")
    print(error)

    print('there')
    if error != '':
        raise RuntimeError(f"Error while reading file: {filepath}. \n {error}")

    if 'Error while loading the script' in output:
        error = output[output.find('Error while loading the script')+len('Error while loading the script:')+1:]
        raise ValueError(f'Error while loading file: {filepath}. \n {error}')
    return output

def load1(filepath):
    """Load spectrum.

    Args:
        filepath (string or pathlib.Path): path to file.

    Returns
        x, y's
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        firstline = f.readline().rstrip()

    if firstline == '#Spectra: 1':
        data = np.genfromtxt(filepath, skip_header=5, usecols=(0, 2))
    else:
        temp = np.loadtxt(filepath, skiprows=5)
        x  = temp[:, 0]
        ys = temp[:, 2::2]
        data = np.zeros((len(x), ys.shape[1]+1))
        data[:, 0] = x
        for i in range(ys.shape[1]):
            data[:, i+1] = ys[:, i]

    # return
    if settings.USE_BRIXS:
        if data.shape[1] > 2:
            ss = settings._br.Spectra()
            for i in range(data.shape[1]):
                if i!=0:
                    ss.append(settings._br.Spectrum(x=data[:, 0], y=data[:, i]))
            return ss
        else:
            return settings._br.Spectrum(x=data[:, 0], y=data[:, 1])
    else:
        return data
    
def load_spectrum(*args):
    if len(args) == 1:
        return load1(args[0])
    else:
        if settings.USE_BRIXS:
            ss = settings._br.Spectra()
            for filepath in args:
                temp = load1(filepath)
                if isinstance(temp, settings._br.Spectra):
                    for s in temp:
                        ss.append(s)
                else:
                    ss.append(temp)
            return ss
        else:
            final = []
            for filepath in args:
                final.append(load1(filepath))
            return final


