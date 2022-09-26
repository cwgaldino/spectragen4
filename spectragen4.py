#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package for spectrum generation via crispy 0.7.3 and Quanty.

TO DO:
( ) test spectragen without BRIXS
( ) Check if broaden works with non-monotonic data.
( ) actually, not sure if broaden works fine. Compare with crispy.
( ) replace() docstring
( ) save_parameters() docstring
( ) Calculation.spectrum() docstring
( ) test quanty() on linux
( ) finish Calculation() docstring
"""

# %% settings ==================================================================
# inside <folderpath2crispy> one must find the folders 'crispy', 'docs', 'scripts', etc...
QUANTY_FILEPATH = r'D:\galdino\Documents\CuSb2O6\analysis\rixs\modeling\v4\quanty\quanty_win\QuantyWin64.exe'
CRISPY_FILEPATH = r'D:\galdino\Documents\CuSb2O6\analysis\rixs\modeling\v4\crispy-0.7.3\crispy-0.7.3'
USE_BRIXS = True

# %% Imports ===================================================================
# standard libraries
from pathlib import Path
import numpy as np
import copy
import os
import subprocess


# specific libraries
from collections.abc import MutableMapping
import tempfile
import shutil

# crispy
CRISPY_FILEPATH = Path(CRISPY_FILEPATH)
import sys
sys.path.append(str(CRISPY_FILEPATH))
from crispy.gui.quanty import QuantyCalculation
from crispy.gui.quanty import broaden

# read crispy default paramenters
import gzip
import json
f = gzip.open(CRISPY_FILEPATH/r'crispy\modules\quanty\parameters\parameters.json.gz', 'r')
temp = json.load(f)
default = temp['elements']
del temp

# Operating system
import platform
system = platform.system().lower()
is_windows = system == 'windows'
is_linux = system == 'linux'
is_mac = system == 'darwin'

# Quanty
QUANTY_FILEPATH = Path(QUANTY_FILEPATH)

# BRIXS
if USE_BRIXS:
    import brixs as br

# %% support classes ===========================================================
class _hamiltonianState(MutableMapping):
    def __init__(self, initial):
        self.store = dict(initial)
        for item in self.store:
            self[item] = self.store[item]

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
        if value:
            self.store[name] = True
        else:
            self.store[name] = False
        self.check_hybridization()

    def __delitem__(self, key):
        raise AttributeError('Itens cannot be deleted')

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len({name:self.store[name] for name in self.store if self.store[name] is not None})

    def check_hybridization(self):
        for item in self.store:
            lmct = False
            mlct = False
            if item.startswith('LMCT'):
                lmct = self.store[item]
            elif item.startswith('MLCT'):
                mlct = self.store[item]
        if lmct and mlct:
            for item in self.store:
                if item.startswith('LMCT') or item.startswith('MLCT'):
                    self.store[item] = False
            raise ValueError('Ligands Hybridization LMCT and MLCT cannot be both True.\nSwitching both to False.')

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
        element (string, optional): transitin metals, lanthanoids and actinoids.
            Default is 'Ni'.
        charge (string, optional): suitable oxidation state of the element as
            '1+', '2+' and so on. Default depends on the element.
        symmetry (string, optional): local symmetry. Possible values are 'Oh',
            'D4h', 'Td', 'C3v', and 'D3h'. Default depends on the element.
        experiment (string, optional): experiment to simulate spectrum. Possible
            values are 'XAS', 'XES', 'XPS', and 'RIXS'. Default is 'XAS'.
        edge (string, optional): investigated edge. Default is 'XAS'.
            * For 'XAS' and 'XPS', possible values are 'K (1s)', 'L1 (2s)', 'L2,3 (2p)', 'M1 (3s)', and 'M2,3 (3p)'.
            * For 'XES', possible values are 'Ka (1s2p)' and 'Kb (1s3p)'.
            * For 'RIXS', possible values are 'K-L2,3 (1s2p)', 'K-M2,3 (1s3p)', and 'L2,3-M4,5 (2p3d)'

        toCalculate (list, optional): type of spectrum to calculate. Default is ['Isotropic']
            * For 'XAS', possible values are 'Circular Dichroism', 'Isotropic', 'Linear Dichroism'.
            * For 'XES' and 'XPS', possible value is 'Isotropic'.
            * For 'RIXS', possible values are 'Isotropic', 'Linear Dichroism'.
            Note: 'Linear Dichroism' for RIXS was manualy implemented (not a crispy feature).
        nPsis (number, optional): number of states to calculate. If None, a suitable
            value will be calculated. default is None.


        magneticField (number, optional): Magnetic field value in Tesla. If zero,
            Crispy will add a very small magnetic field (~1*10^-7 T) in order to
            have nice expected values for observables. To force a zero magnetic
            field you have to turn it off via the ``hamiltonianState``. Default
            is zero (which is modified to a very small magnetic field). Also,
            by default the magnetic field will point in the z direction. That
            can be changed via ``hamiltonianData``.
        temperature (number, optional): temperature in kelvin. Default is 10.

        xLorentzian (tuple, optional):
            * For 'XAS', it is the spectral broadening for the first and second
                half of the spectrum. Default is (0.48, 0.52).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', n.a.
        yLorentzian (tuple, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', energy loss broadening (min value is 0.1). Default is
                (0.1, 0.1). not sure if the tuple works. CHECK

        k1 (tuple, optional): incident wavevector. Default is (0, 0, 1).
        eps11 (tuple, optional): polarization vector. It will be automatically
            modified to be perpendicular to k1. Default depends on 'k1'.
        k2 (tuple, optional): scattered wavevector. Default is (0, 0, 1).
        eps21 = None,       # test galdino. out polarization vector. RIXS n.a.

        xMin (number, optional):
            * For 'XAS', spectrum energy minimum (in eV).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', incident energy minimum (in eV).
            Default is None, where a suitable value depending on the element will
            be used.
        xMax (number, optional):
            * For 'XAS', spectrum energy maximum (in eV).
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', incident energy maximum (in eV).
            Default is None, where a suitable value depending on the element will
            be used.
        xNPoints (number, optional):
            * For 'XAS', number of points in the spectra.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', number of incident energies.
            Default is None, where a suitable value depending on the element will
            be used.
        yMin (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', energy loss minimum (in eV).
            Default is None, where a suitable value depending on the element will
            be used.
        yMax (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', energy loss maximum (in eV).
            Default is None, where a suitable value depending on the element will
            be used.
        yNPoints (number, optional):
            * For 'XAS', n.a.
            * For 'XES' and 'XPS', not sure. Need to check!
            * For 'RIXS', number of points in energy loss.
            Default is None, where a suitable value depending on the element will
            be used.




    Attributes:
        q (quanty calculation object)

        element (string):
        charge (string):
        symmetry (string):
        experiment (string):
        edge (string):

        toCalculate (tuple):
        nPsis (int):
        nConfigurations (int): number of configurations. If None,
            a suitable value will be calculated. Default is None.
            APARENTLY THIS IS ALWAYS 1

        xMin
        xMax
        xNPoints (int):
        yMin
        yMax
        yNPoints (int):

        k1
        eps11
        eps12 (tuple): cannot be set. It is perpendicular to k1 and eps11
        k2
        eps21
        eps22 (tuple): cannot be set. It is perpendicular to k2 and eps21

        temperature
        magneticField

        xLorentzian
        yLorentzian

        hamiltonianState (dict, optional): a dictionary that turns on or off the
            different contributions to the hamiltonian. Default is shuch that
            'Atomic', 'Crystal Field', and 'Magnetic Field' terms will be True.
        hamiltonianData (dict, optional): dictionary with values for the strentgh
            of each different contributions to the hamiltonian. The default value
            is different for each element and charge state. It is recomended to
            create the spectragen.Calculation() object with the default values
            and then edit it from there.


    Methods:
        save()
        load()
        plot()
        imshow()
        binning()
        calculate_histogram()
        calculate_spectrum()
        floor()
        calculate_shifts()
        set_shifts()
        fix_curvature()

    Create input file for Quanty.


    """
    def __init__(self, element='Ni',
                         charge='2+',
                         symmetry=None,
                         experiment=None,
                         edge=None,
                         #
                         toCalculate='Isotropic',
                         nPsis = None,
                         #
                         magneticField=0,
                         temperature=10,
                         #
                         xLorentzian=(0.48, 0.52),
                         yLorentzian=(0.1, 0.1),
                         #
                         quantization_axis = (0, 0, 1),
                         # k1 = (0, 0, 1),
                         eps11 = (1, 0, 0),#None,
                         # k2 = (0, 0, 1),
                         eps21 = (1, 0, 0),#None,
                         #
                         xMin = None,
                         xMax = None,
                         xNPoints = None,
                         yMin = None,
                         yMax = None,
                         yNPoints = None,
                         #
                         baseName = None,
                         ):
        self._epsilon = 10e-14

        # crispy attributes
        self.verbosity = '0x0000'
        self.denseBorder = '2000'

        # primary attributes
        self._set_primary_attributes(element, charge, symmetry, experiment, edge)

        # calculation attributes
        self.toCalculate = toCalculate
        self.nPsis = nPsis
        self.nConfigurations = 1  # this is always 1 in Crispy.
        self.baseName = baseName

        # spectrum attributes
        self.xMin = xMin
        self.xMax = xMax
        self.xNPoints = xNPoints
        self.yMin = yMin
        self.yMax = yMax
        self.yNPoints = yNPoints

        # incidence orientation attributes
        # self._k1 = k1
        self._magneticField = 0
        self.quantization_axis = quantization_axis
        self.magneticField = magneticField  # experiment attributes

        # self.k1 = k1
        # if eps11 is not None:
        self.eps11 = eps11

        # scattered orientation attributes
        # self.k2 = k2
        # if eps21 is not None:
        self.eps21 = eps21

        # experiment attributes
        self.temperature = temperature

        # broadening attributes
        self.xLorentzian = xLorentzian
        self.yLorentzian = yLorentzian

        # # hamiltonian
        self.hamiltonianState = _hamiltonianState(self.q.hamiltonianState)
        self.hamiltonianData  = _hamiltonianData(self.q.hamiltonianData)

    def _set_primary_attributes(self, element, charge, symmetry, experiment, edge):
        """Validate and set primary attributes."""

        # validation ============================
        if element is None:
            element = 'Ni'
        assert element in default, f'Not a valid element.\nValid elements are: {tuple(default.keys())}'

        valid_charges = tuple(default[element]['charges'].keys())
        if charge is None:
            charge = valid_charges[0]
        assert charge in valid_charges, f'Not a valid oxidation state for {element}\nValid oxidation states are: {valid_charges}'

        valid_symmetries = tuple(default[element]['charges'][charge]['symmetries'].keys())
        if symmetry is None:
            symmetry = valid_symmetries[0]
        assert symmetry in valid_symmetries, f'Not a valid symmetry for {element} {charge}\nValid symmetries are: {valid_symmetries}'

        valid_experiments = tuple(default[element]['charges'][charge]['symmetries'][symmetry]['experiments'].keys())
        if experiment is None:
            experiment = valid_experiments[0]
        assert experiment in valid_experiments, f'Not a valid experiment for {element} {charge} {symmetry}\nValid experiments are: {valid_experiments}'

        valid_edges = tuple(default[element]['charges'][charge]['symmetries'][symmetry]['experiments'][experiment]['edges'].keys())
        if edge is None:
            edge = valid_edges[0]
        assert edge in valid_edges, f'Not a valid edge for {element} {charge} {symmetry} {experiment}\nValid edges are: {valid_edges}'

        # initialize calculation object ===========================
        self.q = QuantyCalculation(element=element,
                                   charge=charge,
                                   symmetry=symmetry,
                                   experiment=experiment,
                                   edge=edge)

        # ser attributes ========================================
        self._element    = self.q.element
        self._charge     = self.q.charge
        self._symmetry   = self.q.symmetry
        self._experiment = self.q.experiment
        self._edge       = self.q.edge

        # set verbosity (this is set the same as in crispy)
        self.q.verbosity =  self.verbosity

        # set border (this is set the same as in crispy)
        self.q.denseBorder = self.denseBorder

    # primary attributes =================================
    # element
    @property
    def element(self):
        return self._element
    @element.setter
    def element(self, value):
        # self._initialize(value, self.charge, self.symmetry, self.experiment, self.edge)
        # print(f"Primary parameter changed (element, charge, symmetry, experiment, edge). All parameters set to default.")
        raise AttributeError('Primary atributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new spectragen.Calculation() object')
    @element.deleter
    def element(self):
        raise AttributeError('Cannot delete object.')

    # charge
    @property
    def charge(self):
        return self._charge
    @charge.setter
    def charge(self, value):
        # self._initialize(self.element, value, self.symmetry, self.experiment, self.edge)
        # print(f"Primary parameter changed (element, charge, symmetry, experiment, edge). All parameters set to default.")
        raise AttributeError('Primary atributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new spectragen.Calculation() object')
    @charge.deleter
    def charge(self):
        raise AttributeError('Cannot delete object.')

    # symmetry
    @property
    def symmetry(self):
        return self._symmetry
    @symmetry.setter
    def symmetry(self, value):
        # self._initialize(self.element, self.charge, value, self.experiment, self.edge)
        # print(f"Primary parameter changed (element, charge, symmetry, experiment, edge). All parameters set to default.")
        raise AttributeError('Primary atributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new spectragen.Calculation() object')
    @symmetry.deleter
    def symmetry(self):
        raise AttributeError('Cannot delete object.')

    # experiment
    @property
    def experiment(self):
        return self._experiment
    @experiment.setter
    def experiment(self, value):
        # self._initialize(self.element, self.charge, self.symmetry, value, self.symmetry)
        # print(f"Primary parameter changed (element, charge, symmetry, experiment, edge). All parameters set to default.")
        raise AttributeError('Primary atributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new spectragen.Calculation() object')
    @experiment.deleter
    def experiment(self):
        raise AttributeError('Cannot delete object.')

    # edge
    @property
    def edge(self):
        return self._edge
    @edge.setter
    def edge(self, value):
        # self._initialize(self.element, self.charge, self.symmetry, self.experiment, value)
        # print(f"Primary parameter changed (element, charge, symmetry, experiment, edge). All parameters set to default.")
        raise AttributeError('Primary atributes cannot be changed (element, charge, symmetry, experiment, edge).\nPlease, start a new spectragen.Calculation() object')
    @edge.deleter
    def edge(self):
        raise AttributeError('Cannot delete object.')

    # calculation attributes =================================
    # toCalculate
    @property
    def toCalculate(self):
        return self._toCalculate
    @toCalculate.setter
    def toCalculate(self, value):

        # check value
        error = "Invalid value for toCalculate\nValid values are:\nXAS: 'Circular Dichroism', 'Isotropic', 'Linear Dichroism'\nXES: 'Isotropic'\nXPS: 'Isotropic'\nRIXS: 'Isotropic', 'Linear Dichroism'"
        assert type(value) == str, error
        assert value.lower().startswith(("cir", "cd", "lin", "ld", "iso", 'i')), error
        if self.experiment == 'RIXS':
            assert value.lower().startswith(("lin", "ld", "iso", 'i')), error
        if self.experiment == 'XES' or self.experiment == 'XPS':
            assert value.lower().startswith(("iso", 'i')), error

        # convert value to crispy sintax
        if value.lower().startswith(("cir", "cd")):
            value = 'Circular Dichroism'
        elif value.lower().startswith(("lin", "ld")):
            value = 'Linear Dichroism'
        else:
            value = 'Isotropic'

        # set value
        self.q.spectra.toCalculate = [value]
        self.q.spectra.toCalculateChecked = [value]
        # self._toCalculate = copy.copy(self.q.spectra.toCalculate)
        self._toCalculate = value
    @toCalculate.deleter
    def toCalculate(self):
        raise AttributeError('Cannot delete object.')

    # npsis
    @property
    def nPsis(self):
        return self._nPsis
    @nPsis.setter
    def nPsis(self, value):
        if value is None:
            self.q.nPsisAuto = 1
            value = self.q.nPsisMax
        else:
            assert value > 0, 'The number of states must be larger than zero.'
            assert value <= self.q.nPsisMax, f'The selected number of states exceeds the maximum.\nMaximum {self.q.nPsisMax}'
        self.q.nPsis = value
        self._nPsis = copy.copy(self.q.nPsis)
    @nPsis.deleter
    def nPsis(self):
        raise AttributeError('Cannot delete object.')

    # nConfigurations
    @property
    def nConfigurations(self):
        return self._nConfigurations
    @nConfigurations.setter
    def nConfigurations(self, value):
        if value is None:
            value = self.q.nConfigurationsMax
        else:
            assert value <= self.q.nConfigurationsMax, f'The maximum number of configurations is {self.q.nConfigurationsMax}.'

        self.q.nConfigurations = value
        self._nConfigurations = copy.copy(self.q.nConfigurations)
    @nConfigurations.deleter
    def nConfigurations(self):
        raise AttributeError('Cannot delete object.')

    # baseName
    @property
    def baseName(self):
        return self._baseName
    @baseName.setter
    def baseName(self, value):
        if value is None:
            self._baseName = None
            return
        assert type(value) == str, 'baseName must be a string.'
        self._baseName = value
    @baseName.deleter
    def baseName(self):
        raise AttributeError('Cannot delete object.')

    # spectrum attributes ===============================
    # xMin
    @property
    def xMin(self):
        return self._xMin
    @xMin.setter
    def xMin(self, value):
        if value is None:
            value = self.q.xMin
        self.q.xMin = value
        self._xMin = copy.copy(self.q.xMin)
    @xMin.deleter
    def xMin(self):
        raise AttributeError('Cannot delete object.')

    # xMax
    @property
    def xMax(self):
        return self._xMax
    @xMax.setter
    def xMax(self, value):
        if value is None:
            value = self.q.xMax
        self.q.xMax = value
        self._xMax = copy.copy(self.q.xMax)
    @xMax.deleter
    def xMax(self):
        raise AttributeError('Cannot delete object.')

    # xNPoints
    @property
    def xNPoints(self):
        return self._xNPoints
    @xNPoints.setter
    def xNPoints(self, value):
        if value is None:
            value = self.q.xNPoints
        assert int(value) >= 2, 'xNPoints cannot be less than 2.'
        self.q.xNPoints = int(value-1)
        self._xNPoints = int(value)
    @xNPoints.deleter
    def xNPoints(self):
        raise AttributeError('Cannot delete object.')

    # yMax
    @property
    def yMax(self):
        return self._yMax
    @yMax.setter
    def yMax(self, value):
        if value is None:
            value = self.q.yMax
        self.q.yMax = value
        self._yMax = copy.copy(self.q.yMax)
    @yMax.deleter
    def yMax(self):
        raise AttributeError('Cannot delete object.')

    # yMin
    @property
    def yMin(self):
        return self._yMin
    @yMin.setter
    def yMin(self, value):
        if value is None:
            value = self.q.yMin
        self.q.yMin = value
        self._yMin = copy.copy(self.q.yMin)
    @yMin.deleter
    def yMin(self):
        raise AttributeError('Cannot delete object.')

    # yNPoints
    @property
    def yNPoints(self):
        return self._yNPoints
    @yNPoints.setter
    def yNPoints(self, value):
        if value is None:
            value = self.q.yNPoints
        self.q.yNPoints = value
        self._yNPoints = copy.copy(self.q.yNPoints)
    @yNPoints.deleter
    def yNPoints(self):
        raise AttributeError('Cannot delete object.')

    # # orientation attributes ===========================
    # # k1
    # @property
    # def k1(self):
    #     return self._k1
    # @k1.setter
    # def k1(self, value):
    #     if len(value) != 3:
    #         raise ValueError('k1 must be a vector, like [0, 0, 1].')
    #     else:
    #         # The k1 value should be fine; save it.
    #         k1 = value
    #
    #         # The polarization vector must be corrected.
    #         eps11 = self.q.eps11
    #         # If the wave and polarization vectors are not perpendicular, select a
    #         # new perpendicular vector for the polarization.
    #         if np.dot(np.array(k1), np.array(eps11)) != 0:
    #             if k1[2] != 0 or (-k1[0] - k1[1]) != 0:
    #                 eps11 = (k1[2], k1[2], -k1[0] - k1[1])
    #             else:
    #                 eps11 = (-k1[2] - k1[1], k1[0], k1[0])
    #
    #         # store values
    #         self.q.k1 = k1
    #         self.q.eps11 = eps11
    #
    #         self._k1 = copy.copy(self.q.k1)
    #         self.eps11 = copy.copy(self.q.eps11)
    #
    #         # Update the magnetic field.
    #         self.magneticField = self.magneticField
    # @k1.deleter
    # def k1(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # eps11
    # @property
    # def eps11(self):
    #     return self._eps11
    # @eps11.setter
    # def eps11(self, value):
    #     if len(value) != 3:
    #         raise ValueError('eps11 must be a vector, like [0, 1, 0].')
    #     else:
    #         # The eps11 value should be fine; save it.
    #         eps11 = value
    #
    #         if np.dot(np.array(self.k1), np.array(eps11)) > self._epsilon:
    #             raise ValueError('eps11 must be perpendicular to k1. Adjust k1 first.')
    #
    #         # Generate a second, perpendicular, polarization vector to the plane
    #         # defined by the wave vector and the first polarization vector.
    #         eps12 = np.cross(np.array(eps11), np.array(self.k1))
    #         eps12 = eps12.tolist()
    #
    #         # store values
    #         self.q.eps11 = eps11
    #         self._eps11 = copy.copy(self.q.eps11)
    #         self.q.eps12 = eps12
    #         self._eps12 = copy.copy(self.q.eps12)
    # @eps11.deleter
    # def eps11(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # eps12
    # @property
    # def eps12(self):
    #     return self._eps12
    # @eps12.setter
    # def eps12(self, value):
    #     raise ValueError('Cannot edit eps12 manualy as it is defined by k1 and eps11.')
    # @eps12.deleter
    # def eps12(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # k2
    # @property
    # def k2(self):
    #     return self._k2
    # @k2.setter
    # def k2(self, value):
    #     if len(value) != 3:
    #         raise ValueError('k2 must be a vector, like [0, 0, 1].')
    #     else:
    #         # The k2 value should be fine; save it.
    #         k2 = value
    #
    #         # The polarization vector must be corrected.
    #         eps21 = self.q.eps21
    #         # If the wave and polarization vectors are not perpendicular, select a
    #         # new perpendicular vector for the polarization.
    #         if np.dot(np.array(k2), np.array(eps21)) != 0:
    #             if k2[2] != 0 or (-k2[0] - k2[1]) != 0:
    #                 eps21 = (k2[2], k2[2], -k2[0] - k2[1])
    #             else:
    #                 eps21 = (-k2[2] - k2[1], k2[0], k2[0])
    #
    #         # store values
    #         self.q.k2 = k2
    #         self.q.eps21 = eps21
    #
    #         self._k2 = copy.copy(self.q.k2)
    #         self.eps21 = copy.copy(self.q.eps21)
    # @k2.deleter
    # def k2(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # eps21
    # @property
    # def eps21(self):
    #     return self._eps21
    # @eps21.setter
    # def eps21(self, value):
    #     if len(value) != 3:
    #         raise ValueError('eps21 must be a vector, like [0, 1, 0].')
    #     else:
    #         # The eps21 value should be fine; save it.
    #         eps21 = value
    #
    #         if np.dot(np.array(self.k2), np.array(eps21)) > self._epsilon:
    #             raise ValueError('eps21 must be perpendicular to k2. Adjust k2 first.')
    #
    #         # Generate a second, perpendicular, polarization vector to the plane
    #         # defined by the wave vector and the first polarization vector.
    #         eps22 = np.cross(np.array(eps21), np.array(self.k2))
    #         eps22 = eps22.tolist()
    #
    #         # store values
    #         self.q.eps21 = eps21
    #         self._eps21 = copy.copy(self.q.eps21)
    #         self.q.eps22 = eps22
    #         self._eps22 = copy.copy(self.q.eps22)
    # @eps21.deleter
    # def eps21(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # eps22
    # @property
    # def eps22(self):
    #     return self._eps22
    # @eps22.setter
    # def eps22(self, value):
    #     raise ValueError('Cannot edit eps22 manualy as it is defined by k2 and eps21.')
    # @eps22.deleter
    # def eps22(self):
    #     raise AttributeError('Cannot delete object.')

    # # experiment attributes =======================================
    # # magneticField
    # @property
    # def magneticField(self):
    #     return self._magneticField
    # @magneticField.setter
    # def magneticField(self, value):
    #     TESLA_TO_EV = 5.788e-05
    #
    #     # Normalize the current incident vector.
    #     k1 = np.array(self.k1)
    #     k1 = k1 / np.linalg.norm(k1)
    #
    #     configurations = self.q.hamiltonianData['Magnetic Field']
    #     for configuration in configurations:
    #         parameters = configurations[configuration]
    #         for i, parameter in enumerate(parameters):
    #             value2 = float(value * np.abs(k1[i]) * TESLA_TO_EV)
    #             if abs(value2) == 0.0:
    #                     value2 = 0.0
    #             configurations[configuration][parameter] = value2
    #
    #     self.q.magneticField = value
    #     self._magneticField = copy.copy(self.q.magneticField)
    # @magneticField.deleter
    # def magneticField(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # # temperature
    # @property
    # def temperature(self):
    #     return self._temperature
    # @temperature.setter
    # def temperature(self, value):
    #     assert value >= 0, 'Temperature cannot be negative.'
    #     if value == 0:
    #         print('Temperature = 0\nnPsi set to 1.')
    #         self.nPsis = 1
    #     self.q.temperature = value
    #     self._temperature = copy.copy(self.q.temperature)
    # @temperature.deleter
    # def temperature(self):
    #     raise AttributeError('Cannot delete object.')
    #
    # orientation attributes ===========================
    # quantization axis
    @property
    def quantization_axis(self):
        return self._quantization_axis
    @quantization_axis.setter
    def quantization_axis(self, value):
        if len(value) != 3:
            raise ValueError('quantization_axis must be a vector, like [0, 0, 1].')
        else:
            value = normalize(value)
            # store values
            self.q.k1 = value
            self._quantization_axis = copy.copy(self.q.k1)

            # Update the magnetic field.
            self.magneticField = self.magneticField
    @quantization_axis.deleter
    def quantization_axis(self):
        raise AttributeError('Cannot delete object.')

    # # k1
    # @property
    # def k1(self):
    #     return self._k1
    # @k1.setter
    # def k1(self, value):
    #     if len(value) != 3:
    #         raise ValueError('k1 must be a vector, like [0, 0, 1].')
    #     else:
    #         # The k1 value should be fine; save it.
    #         k1 = value
    #
    #         # The polarization vector must be corrected.
    #         eps11 = self.q.eps11
    #         # If the wave and polarization vectors are not perpendicular, select a
    #         # new perpendicular vector for the polarization.
    #         if np.dot(np.array(k1), np.array(eps11)) != 0:
    #             if k1[2] != 0 or (-k1[0] - k1[1]) != 0:
    #                 eps11 = (k1[2], k1[2], -k1[0] - k1[1])
    #             else:
    #                 eps11 = (-k1[2] - k1[1], k1[0], k1[0])
    #
    #         # store values
    #         self.q.k1 = k1
    #         self.q.eps11 = eps11
    #
    #         self._k1 = copy.copy(self.q.k1)
    #         self.eps11 = copy.copy(self.q.eps11)
    #
    #         # Update the magnetic field.
    #         self.magneticField = self.magneticField
    # @k1.deleter
    # def k1(self):
        raise AttributeError('Cannot delete object.')

    # eps11
    @property
    def eps11(self):
        return self._eps11
    @eps11.setter
    def eps11(self, value):
        if len(value) != 3:
            raise ValueError('eps11 must be a vector, like [0, 1, 0].')
        else:
            # The eps11 value should be fine; save it.
            eps11 = normalize(value)

            # if np.dot(np.array(self.k1), np.array(eps11)) > self._epsilon:
            #     raise ValueError('eps11 must be perpendicular to k1. Adjust k1 first.')

            # # Generate a second, perpendicular, polarization vector to the plane
            # # defined by the wave vector and the first polarization vector.
            # eps12 = np.cross(np.array(eps11), np.array(self.k1))
            # eps12 = eps12.tolist()

            # store values
            self.q.eps11 = eps11
            self._eps11 = copy.copy(self.q.eps11)
            # self.q.eps12 = eps12.tolist()
            # self._eps12 = copy.copy(self.q.eps12)
    @eps11.deleter
    def eps11(self):
        raise AttributeError('Cannot delete object.')

    # # eps12
    # @property
    # def eps12(self):
    #     return self._eps12
    # @eps12.setter
    # def eps12(self, value):
    #     raise ValueError('Cannot edit eps12 manualy as it is defined by k1 and eps11.')
    # @eps12.deleter
    # def eps12(self):
    #     raise AttributeError('Cannot delete object.')

    # # k2
    # @property
    # def k2(self):
    #     return self._k2
    # @k2.setter
    # def k2(self, value):
    #     if len(value) != 3:
    #         raise ValueError('k2 must be a vector, like [0, 0, 1].')
    #     else:
    #         # The k2 value should be fine; save it.
    #         k2 = value
    #
    #         # The polarization vector must be corrected.
    #         eps21 = self.q.eps21
    #         # If the wave and polarization vectors are not perpendicular, select a
    #         # new perpendicular vector for the polarization.
    #         if np.dot(np.array(k2), np.array(eps21)) != 0:
    #             if k2[2] != 0 or (-k2[0] - k2[1]) != 0:
    #                 eps21 = (k2[2], k2[2], -k2[0] - k2[1])
    #             else:
    #                 eps21 = (-k2[2] - k2[1], k2[0], k2[0])
    #
    #         # store values
    #         self.q.k2 = k2
    #         self.q.eps21 = eps21
    #
    #         self._k2 = copy.copy(self.q.k2)
    #         self.eps21 = copy.copy(self.q.eps21)
    # @k2.deleter
    # def k2(self):
        raise AttributeError('Cannot delete object.')

    # eps21
    @property
    def eps21(self):
        return self._eps21
    @eps21.setter
    def eps21(self, value):
        if len(value) != 3:
            raise ValueError('eps21 must be a vector, like [0, 1, 0].')
        else:
            # The eps21 value should be fine; save it.
            eps21 = normalize(value)

            # if np.dot(np.array(self.k2), np.array(eps21)) > self._epsilon:
            #     raise ValueError('eps21 must be perpendicular to k2. Adjust k2 first.')
            #
            # # Generate a second, perpendicular, polarization vector to the plane
            # # defined by the wave vector and the first polarization vector.
            # eps22 = np.cross(np.array(eps21), np.array(self.k2))
            # eps22 = eps22.tolist()

            # store values
            self.q.eps21 = eps21
            self._eps21 = copy.copy(self.q.eps21)
            # self.q.eps22 = eps22
            # self._eps22 = copy.copy(self.q.eps22)
    @eps21.deleter
    def eps21(self):
        raise AttributeError('Cannot delete object.')

    # # eps22
    # @property
    # def eps22(self):
    #     return self._eps22
    # @eps22.setter
    # def eps22(self, value):
    #     raise ValueError('Cannot edit eps22 manualy as it is defined by k2 and eps21.')
    # @eps22.deleter
    # def eps22(self):
        raise AttributeError('Cannot delete object.')

    # experiment attributes =======================================
    # magneticField
    @property
    def magneticField(self):
        return self._magneticField
    @magneticField.setter
    def magneticField(self, value):
        TESLA_TO_EV = 5.788e-05

        quantization_axis = np.array(self.quantization_axis)

        configurations = self.q.hamiltonianData['Magnetic Field']
        for configuration in configurations:
            parameters = configurations[configuration]
            for i, parameter in enumerate(parameters):
                value2 = float(value * np.abs(quantization_axis[i]) * TESLA_TO_EV)
                if abs(value2) == 0.0:
                        value2 = 0.0
                configurations[configuration][parameter] = value2

        self.q.magneticField = value
        self._magneticField = copy.copy(self.q.magneticField)
    @magneticField.deleter
    def magneticField(self):
        raise AttributeError('Cannot delete object.')

    # temperature
    @property
    def temperature(self):
        return self._temperature
    @temperature.setter
    def temperature(self, value):
        assert value >= 0, 'Temperature cannot be negative.'
        if value == 0:
            print('Temperature = 0\nnPsi set to 1.')
            self.nPsis = 1
        self.q.temperature = value
        self._temperature = copy.copy(self.q.temperature)
    @temperature.deleter
    def temperature(self):
        raise AttributeError('Cannot delete object.')

    # broadening attributes ================================
    # xLorentzian
    @property
    def xLorentzian(self):
        return self._xLorentzian
    @xLorentzian.setter
    def xLorentzian(self, value):
        try:
            if value < 0.1:
                raise ValueError('Crispy does not accept value less the 0.1 (CHECK).')
            self.q.xLorentzian = (value, )
        except TypeError:
            if any([v < 0.1 for v in value]):
                raise ValueError('Crispy does not accept value less the 0.1 (CHECK).')
            self.q.xLorentzian = value

        self._xLorentzian = copy.copy(self.q.xLorentzian)
    @xLorentzian.deleter
    def xLorentzian(self):
        raise AttributeError('Cannot delete object.')

    # yLorentzian
    @property
    def yLorentzian(self):
        return self._yLorentzian
    @yLorentzian.setter
    def yLorentzian(self, value):
        try:
            if value < 0.1:
                raise ValueError('Crispy does not accept value less the 0.1 (CHECK).')
            self.q.yLorentzian = (value, )
        except TypeError:
            if any([v < 0.1 for v in value]):
                raise ValueError('Crispy does not accept value less the 0.1 (CHECK).')
            self.q.yLorentzian = value

        self._yLorentzian = copy.copy(self.q.yLorentzian)
    @yLorentzian.deleter
    def yLorentzian(self):
        raise AttributeError('Cannot delete object.')


    # Core methods =============================================================
    def save_input(self, filepath=None):
        """Create and save Quanty input file.

        Args:
            filepath (string or path object, optional): filename or file handle.

        Returns:
            None
        """
        # hamiltonian
        self.q.hamiltonianState = {key:(1 if value else 0) for key, value in self.hamiltonianState.items()}
        self.q.hamiltonianData  = copy.deepcopy(self.hamiltonianData)

        if filepath is None:
            filepath = self.baseName
        if filepath is None:
            filepath = 'Untitled'

        filepath2 = Path(filepath).with_suffix('')
        if is_windows:
            filepath3 = '\\\\\\\\'.join(filepath2.parts)
        else:
            filepath3 = str(filepath2)
        self.q.baseName = str(filepath3)
        self.q.saveInput()
        self.baseName = str(filepath2)

        ### This part is for when you want the basename inside the file to be differente from the filename
        # if self.baseName is None:
        #     if is_windows:
        #         filepath3 = '\\\\\\\\'.join(filepath2.parts)
        #     else:
        #         filepath3 = str(filepath2)
        #
        #     self.q.baseName = str(filepath3)
        #     self.q.saveInput()
        #
        # else:
        #     filepath2 = filepath2.parent / str('aaa'+str(filepath2.name)+'aaa')
        #     if is_windows:
        #         filepath3 = '\\\\\\\\'.join(filepath2.parts)
        #     else:
        #         filepath3 = str(filepath2)
        #
        #     # save temporary file
        #     self.q.baseName = str(filepath3)
        #     self.q.saveInput()
        #
        #     pattern = "G.Print({{'file', '" + filepath3 + "_' .. suffix .. '.spec'}})" + '\n'
        #     subst   = "G.Print({{'file', '" + self.baseName + "_' .. suffix .. '.spec'}})" + '\n'
        #     replace(filepath2.with_suffix('.lua'), pattern, subst)

        # replacements =========================================================
        filepath2 = filepath2.with_suffix('.lua')
        if self.experiment == 'RIXS' and 'Linear Dichroism' in self.toCalculate:
            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    e2 = {" + ','.join([str(x) for x in self.eps21]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    T_3d_2p = e2[1] * Tx_3d_2p + e2[2] * Ty_3d_2p + e2[3] * Tz_3d_2p\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, T_3d_2p, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'DenseBorder', DenseBorder}})"
            replace(filepath2, pattern, subst)

            pattern = "G = G + CreateResonantSpectra(H_m, H_f, {Tx_2p_3d, Ty_2p_3d, Tz_2p_3d}, {Tx_3d_2p, Ty_3d_2p, Tz_3d_2p}, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            subst = "e1 = {" + ','.join([str(x) for x in self.eps11]) + "}\n" + \
                "    e2 = {" + ','.join([str(x) for x in self.eps21]) + "}\n" + \
                "    T_2p_3d = e1[1] * Tx_2p_3d + e1[2] * Ty_2p_3d + e1[3] * Tz_2p_3d\n" + \
                "    T_3d_2p = e2[1] * Tx_3d_2p + e2[2] * Ty_3d_2p + e2[3] * Tz_3d_2p\n" + \
                "    G = G + CreateResonantSpectra(H_m, H_f, T_2p_3d, T_3d_2p, Psis_i, {{'Emin1', Emin1}, {'Emax1', Emax1}, {'NE1', NE1}, {'Gamma1', Gamma1}, {'Emin2', Emin2}, {'Emax2', Emax2}, {'NE2', NE2}, {'Gamma2', Gamma2}, {'restrictions1', CalculationRestrictions}, {'restrictions2', CalculationRestrictions}, {'DenseBorder', DenseBorder}})"
            replace(filepath2, pattern, subst)

            pattern = "    for j = 1, 3 * 3 do"
            subst   = "    for j = 1, 1 do"
            replace(filepath2, pattern, subst)

            pattern = "_iso.spec'}})"
            subst   = "_h.spec'}})"
            replace(filepath2, pattern, subst)

    def run(self, filepath=None):
        """Run Quanty.

        Args:
            QUANTY_FILEPATH (string or pathlib.Path): path to Quanty executable.
            filepath (string or pathlib.Path): path to file.

        Returns:
            String with calculation output (stdout).
        """
        if filepath is None:
            filepath = Path(self.baseName).with_suffix('.lua')
        return quanty(filepath)

    # Spectrum methods =========================================================
    def _load_spectrum_via_brixs(self, filepath):
        ss = load_spectrum(filepath)

        if type(ss) == br.Spectra:
            for s in ss:
                s.hamiltonianData = self.hamiltonianData
                s.magneticField   = self.magneticField
                s.temperature     = self.temperature
                s.xLorentzian     = self.xLorentzian
                s.yLorentzian     = self.yLorentzian
                s.quantization_axis = self.quantization_axis
                # s.k1              = self.k1
                s.eps11           = self.eps11
                # s.eps12           = self.eps12
                # s.k2              = self.k2
                s.eps21           = self.eps21
                s.xMin            = self.xMin
                s.xMax            = self.xMax
                s.xNPoints        = self.xNPoints
                s.yMin            = self.yMin
                s.yMax            = self.yMax
                s.yNPoints        = self.yNPoints

            if self.experiment == 'RIXS':
                incident_energies = np.linspace(self.xMin, self.xMax, self.xNPoints)
                for i in range(len(ss)):
                    ss[i].incident_energy = incident_energies[i]

        ss.hamiltonianData = self.hamiltonianData
        ss.magneticField   = self.magneticField
        ss.temperature     = self.temperature
        ss.xLorentzian     = self.xLorentzian
        ss.yLorentzian     = self.yLorentzian
        ss.quantization_axis = self.quantization_axis
        # ss.k1              = self.k1
        ss.eps11           = self.eps11
        # ss.eps12           = self.eps12
        # ss.k2              = self.k2
        ss.eps21           = self.eps21
        ss.xMin            = self.xMin
        ss.xMax            = self.xMax
        ss.xNPoints        = self.xNPoints
        ss.yMin            = self.yMin
        ss.yMax            = self.yMax
        ss.yNPoints        = self.yNPoints

        return ss

    def _get_spectrum_via_brixs(self, filepath=None):

        if filepath is None:
            filepath = Path(self.baseName).with_suffix('.spec')

            if self.toCalculate == 'Isotropic':
                filepath = filepath.parent / str(filepath.name).replace('.spec', '_iso.spec')
                return self._load_spectrum_via_brixs(filepath)

            elif self.toCalculate == 'Circular Dichroism':
                filepath = filepath.parent / str(filepath.name).replace('.spec', '_l.spec')
                s1 = self._load_spectrum_via_brixs(filepath)
                s1.polarization = 'left'

                filepath = filepath.parent / str(filepath.name).replace('.spec', '_r.spec')
                s2 = self._load_spectrum_via_brixs(filepath)
                s2.polarization = 'right'
                return s1, s2

            elif self.toCalculate == 'Linear Dichroism':
                filepath = filepath.parent / str(filepath.name).replace('.spec', '_h.spec')
                s1 = self._load_spectrum_via_brixs(filepath)
                s1.polarization = 'horizontal'

                if self.experiment == 'RIXS':
                    s2 = None
                    s1.incident_energy = [s.incident_energy for s in s1]
                else:
                    filepath = filepath.parent / str(filepath.name).replace('.spec', '_v.spec')
                    s2 = self._load_spectrum_via_brixs(filepath)
                    s2.polarization = 'vertical'
                return s1, s2

        else:
            return self._load_spectrum_via_brixs(filepath)

    def spectrum(self, filepath=None):
        if USE_BRIXS:
            return self._get_spectrum_via_brixs()
        else:
            if filepath is None:
                filepath = Path(self.baseName).with_suffix('.spec')

                if self.toCalculate == 'Isotropic':
                    filepath = filepath.parent / str(filepath.name).replace('.spec', '_iso.spec')
                    return load_spectrum(filepath)

                elif self.toCalculate == 'Circular Dichroism':
                    filepath = filepath.parent / str(filepath.name).replace('.spec', '_l.spec')
                    x, y_l = load_spectrum(filepath)

                    filepath = filepath.parent / str(filepath.name).replace('.spec', '_r.spec')
                    x, y_r = load_spectrum(filepath)
                    return x, y_l, y_r

                elif self.toCalculate == 'Linear Dichroism':
                    filepath = filepath.parent / str(filepath.name).replace('.spec', '_h.spec')
                    x, y_h = load_spectrum(filepath)

                    if self.experiment == 'RIXS':
                        pass
                    else:
                        filepath = filepath.parent / str(filepath.name).replace('.spec', '_v.spec')
                        x, y_v = load_spectrum(filepath)
                    return x, y_h, y_v
            else:
                return load_spectrum(filepath)

    # parameter methods ========================================================
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
                      , quantization_axis = self.quantization_axis
                      # ,k1               = self.k1
                      ,eps11            = self.eps11
                      # ,k2               = self.k2
                      ,eps21            = self.eps21
                      #
                      ,xMin             = self.xMin
                      ,xMax             = self.xMax
                      ,xNPoints         = self.xNPoints
                      ,yMin             = self.yMin
                      ,yMax             = self.yMax
                      ,yNPoints         = self.yNPoints
                      #
                      ,baseName        = self.baseName
                      #
                      ,hamiltonianState = dict(self.hamiltonianState)
                      ,hamiltonianData  = p
                      )

    def save_parameters(self, filepath=None):
        """Save calculation parameters in a text file.

        Args:
            filepath (string or path object, optional): filename or file handle.

        Returns:
            None

        See Also:
            :py:func:`load_parameters`
        """
        if filepath is None:
            filepath = self.baseName
        if filepath is None:
            filepath = 'Untitled'
        filepath = str(Path(filepath).with_suffix('.par'))
        _save_obj(obj=self.get_parameters(), filepath=filepath)

    def load_parameters(self, filepath=None):
        """Load calculation parameters.

        Args:
            filepath (string or pathlib.Path): filepath.

        Returns:
            None

        See Also:
            :py:func:`save_parameters`
        """
        if filepath is None:
            filepath = self.baseName
        if filepath is None:
            raise ValueError('filepath must be given.')
        filepath = str(Path(filepath).with_suffix('.par'))
        par = _load_obj(filepath)
        hamiltonianData = par.pop('hamiltonianData')
        hamiltonianState = par.pop('hamiltonianState')
        # print(par)
        self.__init__(**par)
        for k1 in hamiltonianState:
            self.hamiltonianState[k1] = hamiltonianState[k1]
            # print(hamiltonianState[k1])
        for k1 in hamiltonianData:
            for k2 in hamiltonianData[k1]:
                for k3 in hamiltonianData[k1][k2]:
                    self.hamiltonianData[k1][k2][k3] = hamiltonianData[k1][k2][k3]


# %% support functions =========================================================
def replace(filepath, pattern, subst):
    """replace string in a file by another string."""
    # Create temp file
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        with open(filepath) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    # Copy the file permissions from the old file to the new file
    shutil.copymode(filepath, abs_path)
    # Remove original file
    os.remove(filepath)
    # Move new file
    shutil.move(abs_path, filepath)

def quanty(filepath):
    """Run Quanty.

    Args:
        filepath (string or pathlib.Path): path to file.

    Returns:
        Calculation output (stdout).
    """
    quanty_exe = Path(QUANTY_FILEPATH)
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
    output = quanty.stdout.read().decode("utf-8")
    error  = quanty.stderr.read().decode("utf-8")

    if error != '':
        raise RuntimeError(f"Error while reading file: {filepath}. \n {error}")

    if 'Error while loading the script' in output:
        error = output[output.find('Error while loading the script')+len('Error while loading the script:')+1:]
        raise ValueError(f'Error while loading file: {filepath}. \n {error}')
    return output

def load_spectrum(filepath):
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
        data = np.genfromtxt(filepath, skip_header=5)
        if USE_BRIXS:
            return br.Spectrum(x=data[:, 0], y=data[:, 2])
        else:
            return data[:, 0], data[:, 2]
    else:
        data = np.loadtxt(filepath, skiprows=5)
        x = data[:, 0]
        ys = data[:, 2::2]
        ys = {i:ys[:, i] for i in range(0, len(ys[0, :]+1))}
        if USE_BRIXS:
            ss = br.Spectra(n=len(ys))
            for i in range(len(ys)):
                ss[i] = br.Spectrum(x=x, y=ys[i])
            return ss
        else:
            return x, ys

def normalize(vector):
    """Returns nomalized vector."""
    if np.linalg.norm(vector) != 1:
        return vector/np.linalg.norm(vector)
    else:
        return vector

def _save_obj(obj, filepath='./Untitled.txt', check_overwrite=False, pretty_print=True):
    """Save object (array, dictionary, list, etc...) in a txt file.

    Args:
        obj (object): object to be saved.
        filepath (str or pathlib.Path, optional): path to save file.
        check_overwrite (bool, optional): if True, it will check if file exists
            and ask if user want to overwrite file.

    Returns:
        None

    See Also:
        :py:func:`load_obj`
    """
    filepath = Path(filepath)

    if check_overwrite:
        if filepath.exists() == True:
            if filepath.is_file() == True:
                if query('File already exists!! Do you wish to ovewrite it?', 'yes') == True:
                    pass
                else:
                    warnings.warn('File not saved because user did not allow overwriting.')
                    return
            else:
                warnings.warn('filepath is pointing to a folder. Saving file as Untitled.txt')
                filepath = filepath/'Untitled.txt'

    with open(str(filepath), 'w') as file:
        if pretty_print:
            file.write(json.dumps(obj, indent=4, sort_keys=False))
        else:
            file.write(json.dumps(obj))

def _load_obj(filepath):
    """Load object (array, dictionary, list, etc...) from a txt file.

    Args:
        filepath (str or pathlib.Path): file path to load.

    Returns:
        object.

    See Also:
        :py:func:`save_obj`
    """
    filepath = Path(filepath)

    with open(str(filepath), 'r') as file:
        # if dict_keys_to_int:
        #     obj = json.load(file, object_hook=_to_int)
        # else:
        obj = json.load(file)
    return obj

def broaden(self, x, y, value):
    """Apply gaussian broadening to spectrum.

    Args:
        value (number): fwhm value of the gaussian broadening.

    Returns:
        Broadened y
    """
    fwhm = value
    xScale = np.abs(min(x) - max(x)) / x.shape[0]
    fwhm = value/xScale
    y = broaden(y, fwhm, 'gaussian')
    return y

if USE_BRIXS:
    def _broaden(self, value):
        """Apply gaussian broadening to spectrum.

        Args:
            value (number): fwhm value of the gaussian broadening.

        Returns:
            None
        """
        # try: xMin = self.xMin
        # except AttributeError: xMin = min(self.x)
        # try: xMax = self.xMax
        # except AttributeError: xMax = max(self.x)
        # try: xNPoints = self.xNPoints
        # except AttributeError: xNPoints = len(self.x)
        # x2 = np.linspace(xMin, xMax, xNPoints)
        # y2 = self.y.flatten()

        # fwhm = value
        xScale = np.abs(self.x.min() - self.x.max()) / self.x.shape[0]
        fwhm = value/xScale
        self.y = broaden(self.y, fwhm, 'gaussian')

    br.Spectrum.broaden = _broaden
