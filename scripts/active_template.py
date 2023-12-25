"""
Example script for active learning workflow for Ir0.5Mo0.5O2, change
composition with BASE_OXIDE, STOICHIOMETRY, and DOPANT_METAL variables.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
import scipy
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from utils import change_composition_doped

BASE_OXIDE = 'iro2' # or 'iro3'
STOICHIOMETRY = 'ab2' # or 'ab3'
DOPANT_METAL = 'Mo' # or other dopant metal

np.random.seed(0)
STOPPING = []

KERNEL = (1 * RBF(length_scale=np.full((3, ), 1).tolist()) +
          .5 * RBF(length_scale=np.full((3, ), 0.5).tolist()) +
          WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-10, 1e+1)))

with open(DOPANT_METAL + '_chemsys.pckl', 'rb') as chemsys:
    REFERENCE_ENERGIES = pickle.load(chemsys)

with open('doped_'+ STOICHIOMETRY + '.pkl', 'rb') as strucutres:
    AB2 = pickle.load(strucutres)

NEW_STRUCTURES = []
for i, structure in enumerate(AB2):
    NEW_STRUCTURES1 = change_composition_doped([structure[0]], 'Ir',
                                               DOPANT_METAL, 'O')
    for j, new_structure in enumerate(NEW_STRUCTURES1):
        NEW_STRUCTURES.append(new_structure)

POOL = NEW_STRUCTURES

POOL = np.asarray(POOL)
for structure in POOL:
    structure.scale_lattice(11 * len(structure._sites))

DATA = pd.read_csv(DOPANT_METAL + '_doped_' + BASE_OXIDE + '_features.csv',
                   index_col='mp-ids')
DATA = DATA.dropna()
VARIANCE_TRANSFORM = VarianceThreshold(threshold=0).fit(DATA)
DATA = DATA.iloc[:, VARIANCE_TRANSFORM.get_support(indices=True)]

PC = PCA(n_components=10).fit_transform(DATA)

PC_DATA = pd.DataFrame()
PC_DATA['index'] = DATA.index
PC_DATA = PC_DATA.set_index('index')
for i in range(3):
    PC_DATA['PC_{}'.format(i)] = PC[:, i]
X_TRAIN, X_TEST, POOL_TRAIN, POOL_TEST = train_test_split(PC_DATA, POOL,
                                                          test_size=5)
os.chdir(DOPANT_METAL)
NEW_ENERGIES = []
NOT_CONVERGED = []
# Study first 5 random materials
for i in range(5):
    try:
        os.mkdir(X_TEST.index[i])
        os.chdir(X_TEST.index[i])
    except FileExistsError:
        os.chdir(X_TEST.index[i])
    print(X_TEST.index[i])
    try:
        computed_entry = (Vasprun('vasprun.xml').
                          get_computed_entry(inc_structure=False))
        with open('conv.pkl', 'rb') as ckpt:
            conv = pickle.load(ckpt)
        if conv != 6:
            raise ValueError('Not converged!')
        if Vasprun('vasprun.xml').converged is False:
            raise ValueError('Not converged!')
        pd_entry = PDEntry(computed_entry.composition,
                           computed_entry.energy)
        NEW_ENERGIES.append(PhaseDiagram(REFERENCE_ENERGIES).
                            get_form_energy_per_atom(pd_entry))
    except:
        with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                  '_unconverged.pckl', 'rb') as strucutres:
            NOT_CONVERGED_GLOBAL = pickle.load(strucutres)
        if X_TEST.index[i] in NOT_CONVERGED_GLOBAL:
            NOT_CONVERGED.append(X_TEST.index[i])
        else:
            try:
                with open('conv.pkl', 'rb') as ckpt:
                    conv = pickle.load(ckpt)
                (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                            user_incar_settings={'NCORE': 4}).
                 write_input(os.getcwd()))

            except:
                conv = 0

            if conv == 0:
                (MPRelaxSet(POOL_TEST[i],
                            user_incar_settings={'ISIF': 6,
                                                 'NCORE': 4}).
                 write_input(os.getcwd()))
                os.system('mpirun -n 20 vasp_std')
                conv += 1
                with open('conv.pkl', 'wb') as ckpt:
                    ab3 = pickle.dump(conv, ckpt)
            if conv == 1:
                try:
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                except ValueError:
                    conv = 6
            if conv == 2:
                try:
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                except ValueError:
                    conv = 6
            if conv == 3:
                try:
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                except ValueError:
                    conv = 6
            if conv == 4:
                try:
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                except ValueError:
                    conv = 6
            if conv == 5:
                try:
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NSW': 0,
                                                     'IBRION': -1,
                                                     'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                except ValueError:
                    conv = 6
            if conv == 6:
                try:
                    if Vasprun('vasprun.xml').converged is True:
                        pass
                    else:
                        try:
                            (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                        user_incar_settings={'NSW': 0,
                                                             'IBRION': -1,
                                                             'NCORE': 4}).
                             write_input(os.getcwd()))
                            os.system('mpirun -n 20 vasp_std')
                            conv += 1
                            with open('conv.pkl', 'wb') as ckpt:
                                ab3 = pickle.dump(conv, ckpt)
                        except ValueError:
                            pass
                except:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NSW': 0,
                                                         'IBRION': -1,
                                                         'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        pass
            try:
                computed_entry = (Vasprun('vasprun.xml').
                                  get_computed_entry(inc_structure=True))
                if Vasprun('vasprun.xml').converged is False:
                    NOT_CONVERGED.append(X_TEST.index[i])
                    NOT_CONVERGED_GLOBAL.append(X_TEST.index[i])
                    with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                              '_unconverged.pckl', 'wb') as strucutres:
                        ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)
                elif Vasprun('vasprun.xml').final_energy > 0:
                    NOT_CONVERGED.append(X_TEST.index[i])
                    NOT_CONVERGED_GLOBAL.append(X_TEST.index[i])
                    with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                              '_unconverged.pckl', 'wb') as strucutres:
                        ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)
                else:
                    pd_entry = PDEntry(computed_entry.composition,
                                       computed_entry.energy)
                    NEW_ENERGIES.append(PhaseDiagram(REFERENCE_ENERGIES).
                                        get_form_energy_per_atom(pd_entry))

            except:

                NOT_CONVERGED.append(X_TEST.index[i])
                NOT_CONVERGED_GLOBAL.append(X_TEST.index[i])
                with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                          '_unconverged.pckl', 'wb') as strucutres:
                    ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)

    os.chdir('..')
# retrain model and calculate EI
X_TEST = X_TEST.drop(index=NOT_CONVERGED, axis=0)
X_TEST = X_TEST.copy()
X_TEST['formation_energy'] = np.array([NEW_ENERGIES]).flatten()
GP = GaussianProcessRegressor(kernel=KERNEL,
                              normalize_y=True, n_restarts_optimizer=1000)
GP.fit(X_TEST.loc[:, X_TEST.columns != 'formation_energy'].values,
       X_TEST['formation_energy'].values.reshape(-1, 1))
print(GP.kernel_)
Y_MEAN, Y_STD = GP.predict(X_TRAIN.values, return_std=True)
Y_MEAN = Y_MEAN.flatten()
print(np.amin(Y_MEAN), np.amax(Y_MEAN))
print(np.amin(Y_STD), np.amax(Y_STD))
print(np.sort(Y_MEAN-Y_STD)[:5])
Z = (np.min(X_TEST['formation_energy'].values) - Y_MEAN)/Y_STD
EI = ((np.min(X_TEST['formation_energy'].values) - Y_MEAN) *
      scipy.stats.norm.cdf(Z) + Y_STD*scipy.stats.norm.pdf(Z))
NEXT_SYSTEMS = X_TRAIN.iloc[np.argsort(EI)[-1:], :]
POOL_NEXT_SYSTEMS = POOL_TRAIN[np.argsort(EI)[-1:]]
X_TRAIN = X_TRAIN.drop(X_TRAIN.index[np.argsort(EI)[-1:]])
POOL_TRAIN = np.delete(POOL_TRAIN, np.argsort(EI)[-1:])
# study points with highest EI
for _ in range(200):
    NEW_ENERGIES = []
    NOT_CONVERGED = []
    for i in range(1):
        print(NEXT_SYSTEMS.index[i])
        try:
            os.mkdir(NEXT_SYSTEMS.index[i])
            os.chdir(NEXT_SYSTEMS.index[i])
        except FileExistsError:
            os.chdir(NEXT_SYSTEMS.index[i])
        try:
            computed_entry = (Vasprun('vasprun.xml').
                              get_computed_entry(inc_structure=False))
            with open('conv.pkl', 'rb') as ckpt:
                conv = pickle.load(ckpt)
            if conv != 6:
                raise ValueError('Not converged!')
            if Vasprun('vasprun.xml').converged is False:
                raise ValueError('Not converged!')
            pd_entry = PDEntry(computed_entry.composition,
                               computed_entry.energy)
            NEW_ENERGIES.append(PhaseDiagram(REFERENCE_ENERGIES).
                                get_form_energy_per_atom(pd_entry))
        except:
            with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                      '_unconverged.pckl', 'rb') as strucutres:
                NOT_CONVERGED_GLOBAL = pickle.load(strucutres)
            if NEXT_SYSTEMS.index[i] in NOT_CONVERGED_GLOBAL:
                NOT_CONVERGED.append(NEXT_SYSTEMS.index[i])
            else:
                try:
                    with open('conv.pkl', 'rb') as ckpt:
                        conv = pickle.load(ckpt)
                    (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                user_incar_settings={'NCORE': 4}).
                     write_input(os.getcwd()))
                except:
                    conv = 0

                if conv == 0:
                    (MPRelaxSet(POOL_NEXT_SYSTEMS[i],
                                user_incar_settings={'ISIF': 6,
                                                     'NCORE': 4}).
                     write_input(os.getcwd()))
                    os.system('mpirun -n 20 vasp_std')
                    conv += 1
                    with open('conv.pkl', 'wb') as ckpt:
                        ab3 = pickle.dump(conv, ckpt)
                if conv == 1:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        conv = 6
                if conv == 2:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        conv = 6
                if conv == 3:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        conv = 6
                if conv == 4:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        conv = 6
                if conv == 5:
                    try:
                        (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                    user_incar_settings={'NSW': 0,
                                                         'IBRION': -1,
                                                         'NCORE': 4}).
                         write_input(os.getcwd()))
                        os.system('mpirun -n 20 vasp_std')
                        conv += 1
                        with open('conv.pkl', 'wb') as ckpt:
                            ab3 = pickle.dump(conv, ckpt)
                    except ValueError:
                        conv = 6
                if conv == 6:
                    try:
                        if Vasprun('vasprun.xml').converged is True:
                            pass
                        else:
                            try:
                                (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                            user_incar_settings={'NSW': 0,
                                                                 'IBRION': -1,
                                                                 'NCORE': 4}).
                                 write_input(os.getcwd()))
                                os.system('mpirun -n 20 vasp_std')
                                conv += 1
                                with open('conv.pkl', 'wb') as ckpt:
                                    ab3 = pickle.dump(conv, ckpt)
                            except ValueError:
                                pass
                    except:
                        try:
                            (MPRelaxSet(Poscar.from_file('CONTCAR').structure,
                                        user_incar_settings={'NSW': 0,
                                                             'IBRION': -1,
                                                             'NCORE': 4}).
                             write_input(os.getcwd()))
                            os.system('mpirun -n 20 vasp_std')
                            conv += 1
                            with open('conv.pkl', 'wb') as ckpt:
                                ab3 = pickle.dump(conv, ckpt)
                        except ValueError:
                            pass
                try:
                    computed_entry = (Vasprun('vasprun.xml').
                                      get_computed_entry(inc_structure=False))
                    if Vasprun('vasprun.xml').converged is False:
                        NOT_CONVERGED.append(NEXT_SYSTEMS.index[i])
                        NOT_CONVERGED_GLOBAL.append(NEXT_SYSTEMS.index[i])
                        with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                                  '_unconverged.pckl', 'wb') as strucutres:
                            ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)
                    elif Vasprun('vasprun.xml').final_energy > 0:
                        NOT_CONVERGED.append(NEXT_SYSTEMS.index[i])
                        NOT_CONVERGED_GLOBAL.append(NEXT_SYSTEMS.index[i])
                        with open('../../' + DOPANT_METAL + '_'+ STOICHIOMETRY +
                                  '_unconverged.pckl', 'wb') as strucutres:
                            ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)
                    else:
                        pd_entry = PDEntry(computed_entry.composition,
                                           computed_entry.energy)
                        NEW_ENERGIES.append(PhaseDiagram(REFERENCE_ENERGIES).
                                            get_form_energy_per_atom(pd_entry))
                except:
                    NOT_CONVERGED.append(NEXT_SYSTEMS.index[i])
                    NOT_CONVERGED_GLOBAL.append(NEXT_SYSTEMS.index[i])
                    with open('../../'+ DOPANT_METAL + '_'+ STOICHIOMETRY +
                              '_unconverged.pckl', 'wb') as strucutres:
                        ab3 = pickle.dump(NOT_CONVERGED_GLOBAL, strucutres)
        os.chdir('..')
    # retrain model and calculate EI
    NEXT_SYSTEMS = NEXT_SYSTEMS.drop(index=NOT_CONVERGED, axis=0)
    NEXT_SYSTEMS = NEXT_SYSTEMS.copy()
    try:
        NEXT_SYSTEMS['formation_energy'] = np.array([NEW_ENERGIES]).flatten()
        X_TEST = pd.concat([X_TEST, NEXT_SYSTEMS])
    except:
        pass
    GP = GaussianProcessRegressor(kernel=KERNEL,
                                  normalize_y=True, n_restarts_optimizer=1000)
    GP.fit(X_TEST.loc[:, X_TEST.columns != 'formation_energy'].values,
           X_TEST['formation_energy'].values.reshape(-1, 1))
    y_test, y_test_std = GP.predict(X_TEST.
                                    loc[:,
                                        X_TEST.columns != 'formation_energy'].
                                    values, return_std=True)
    Y_MEAN, Y_STD = GP.predict(X_TRAIN.values, return_std=True)
    Y_MEAN = Y_MEAN.flatten()
    Z = (np.min(X_TEST['formation_energy'].values) - Y_MEAN)/Y_STD
    EI = ((np.min(X_TEST['formation_energy'].values) - Y_MEAN) *
          scipy.stats.norm.cdf(Z) + Y_STD*scipy.stats.norm.pdf(Z))
    NEXT_SYSTEMS = X_TRAIN.iloc[np.argsort(EI)[-1:], :]
    POOL_NEXT_SYSTEMS = POOL_TRAIN[np.argsort(EI)[-1:]]
    X_TRAIN = X_TRAIN.drop(X_TRAIN.index[np.argsort(EI)[-1:]])
    POOL_TRAIN = np.delete(POOL_TRAIN, np.argsort(EI)[-1:])
    STOPPING.append(np.sort(EI)[-1:] /
                    -np.min(X_TEST['formation_energy'].values))
    # write out convergence stats
    with open(DOPANT_METAL + '_' + BASE_OXIDE + '_active_stopping.txt',
              'w') as f:
        for item in STOPPING:
            f.write("{}\n".format(item))
    X_TEST.to_csv(DOPANT_METAL + '_' + BASE_OXIDE + '_active.csv')
    # check convergence
    if (len(X_TEST['formation_energy'].values) >= 50 and
            (np.sort(EI)[-1:] /
             -np.min(X_TEST['formation_energy'].values)) <= 0.01):
        break
