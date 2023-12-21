"""Make datasets for doped IrO2 materials.
"""
import pickle
import numpy as np
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from utils import change_composition_doped, voro_fingerprint_doped

TMS = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Y', 'Zr', 'Nb',
       'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
       'Au']

POOL = []
mpr = MPRester('API key')
anon_formula = {'A': 1, 'B': 1, 'C': 4}
data = mpr.query({"anonymous_formula": anon_formula},
                 properties=["task_id", "pretty_formula", "structure"])

with open('doped_ab2.pkl', 'rb') as strucutres:
    AB2 = pickle.load(strucutres)


MP_IDS = []
new_count = 0
old_count = 0

for structure in AB2:
    new_count = 0
    for mp_entry in data:
        new_count += 1
        if (StructureMatcher().fit(structure[0], mp_entry['structure'])):
            print(new_count)
            MP_IDS.append(mp_entry['task_id'])
            old_count = new_count
            new_count = 0
            break
print(len(MP_IDS))
print(len(AB2))
for TM in TMS:
    new_mpids = []
    new_structures = []
    for i, structure in enumerate(AB2):
        new_structures1 = change_composition_doped([structure[0]], 'Ir', TM, 'O')
        for j, new_structure in enumerate(new_structures1):
            new_structures.append(new_structure)
            new_mpids.append(MP_IDS[i] + '_{}'.format(j))

    POOL = new_structures

    POOL = np.asarray(POOL)
    for structure in POOL:
        structure.scale_lattice(11 * len(structure._sites))

    data = voro_fingerprint_doped(POOL)
    data['mp-ids'] = np.asarray(new_mpids)
    data = data.set_index('mp-ids')
    data.to_csv(TM + '_doped_iro2_features.csv')
