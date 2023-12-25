"""Helper functions for active learning workflows.
"""
import re
import numpy as np
from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.standard_transformations import (
    SubstitutionTransformation)



def change_composition(structures, a_atom, b_atom):
    """Change anonymous ABx structures to target composition.

    Args:
        structures (list): List containing ABx structures.
        a_atom (str): Species replacing A in target composition.
        b_atom (str): Species replacing B in target composition.

    Returns:
        list: List containing modified strucutres.

    """
    for ind, structure in enumerate(structures):
        atoms = ''.join([i for i in structure.formula if not i.isdigit()])
        numbers = ''.join([i for i in structure.formula if not i.isalpha()])
        numbers.replace(' ', '')
        numbers = numbers.split()
        for i, number in enumerate(numbers):
            numbers[i] = int(number)
        atoms = atoms.replace(' ', '')
        atoms = re.findall('[A-Z][^A-Z]*', atoms)
        if numbers[0] < numbers[1]:
            structures[ind] = (SubstitutionTransformation({atoms[0]: a_atom,
                                                           atoms[1]: b_atom}).
                               apply_transformation(structure))
        else:
            structures[ind] = (SubstitutionTransformation({atoms[1]: a_atom,
                                                           atoms[0]: b_atom}).
                               apply_transformation(structure))
    return structures


def change_composition_doped(structures, a_atom, b_atom, c_atom):
    """Change anonymous ABx structures to target composition.

    Args:
        structures (list): List containing ABx structures.
        a_atom (str): Species replacing A in target composition.
        b_atom (str): Species replacing B in target composition.

    Returns:
        list: List containing modified strucutres.

    """
    new_structures = []
    for ind, structure in enumerate(structures):
        atoms = ''.join([i for i in structure.formula if not i.isdigit()])
        numbers = ''.join([i for i in structure.formula if not i.isalpha()])
        numbers.replace(' ', '')
        numbers = numbers.split()
        for i, number in enumerate(numbers):
            numbers[i] = int(number)
        numbers = np.asarray(numbers)
        atoms = atoms.replace(' ', '')
        atoms = re.findall('[A-Z][^A-Z]*', atoms)

        new_structure1 = (SubstitutionTransformation({atoms[np.argsort(numbers)[0]]: a_atom,
                                                           atoms[np.argsort(numbers)[1]]: b_atom,
                                                           atoms[np.argsort(numbers)[2]]: c_atom}).
                               apply_transformation(structure))
        new_structure2 = (SubstitutionTransformation({atoms[np.argsort(numbers)[0]]: b_atom,
                                                           atoms[np.argsort(numbers)[1]]: a_atom,
                                                           atoms[np.argsort(numbers)[2]]: c_atom}).
                               apply_transformation(structure))
        if StructureMatcher().fit(struct1=new_structure1,
                                  struct2=new_structure2):
            new_structures.append(new_structure1)
        else:
            new_structures.append(new_structure1)
            new_structures.append(new_structure2)




    return new_structures


def voro_fingerprint(structures):
    for i, structure in enumerate(structures):
        structures[i] = AseAtomsAdaptor.get_atoms(structure[0])
    return VoronoiFingerprintGenerator(structures).generate()


def voro_fingerprint_doped(structures):
    for i, structure in enumerate(structures):
        structures[i] = AseAtomsAdaptor.get_atoms(structure)
    return VoronoiFingerprintGenerator(structures).generate()
