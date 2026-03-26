import numpy as np
import pandas as pd
import os
import json
import logging
from pathlib import Path

from ase import Atom, Atoms
from ase.io import read
from ase.calculators.vasp import Vasp

from bo import BayesianOptimization
from data_loader import load_train

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from model_selector2 import ModelSelectorMLE

def check_improvement(y_train, best_energy_so_far, tolerance=1e-4):
    """Check if there's been improvement in the best energy"""
    current_best = y_train.min()
    improvement = best_energy_so_far - current_best
    return current_best, improvement > tolerance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

vasp_params = {
    'xc': 'PBE',
    'algo': 'Normal',
    'ediff': 1e-5,
    'ediffg': -0.025,
    'enaug': 700.0,
    'encut': 400,
    'ibrion': 2,
    'isif': 2,
    'ismear': 0,
    'ispin': 1,
    'istart': 0,
    'isym': 0,
    'ivdw': 12,
    'kgamma': True,
    'lplane': True,
    'lreal': 'Auto',
    'nbands': 148,
    'nelm': 50,
    'npar': 1,
    'nsw': 1,
    'potim': 0.25,
    'prec': 'Normal',
    'sigma': 0.1
}

with open('params.json', 'r') as f:
    config = json.load(f)

train_file_csv = config['train_file_csv']
host_file = config['host_file']
params_file = config['params_file']
host_energy = config['host_energy']
int_atom = config['int_atom']
int_atom_energy = config['int_atom_energy']
batch_size = config['batch_size']
strategy = config['strategy']
N = config['N']

X_init, y_init = load_train(train_file_csv)


atoms_host = read(host_file)
host_positions = atoms_host.get_positions()
rprimd = atoms_host.get_cell()


k = X_init.shape[1] // 3
if X_init.shape[1] % 3 != 0:
    raise ValueError("The number of features is not a multiple of 3. Check the format of training set.")

bo = BayesianOptimization(
    int_atom=int_atom,
    n_candidates=500,
    rmin=1.45,
    rmin_insrt=1.34,
    host=host_positions,
    rprimd=rprimd,
    random_state=42
)

kernel = RBF(
            length_scale=1.0,
            length_scale_bounds=(0.1, 10.0)
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1.0))


X_train, y_train = X_init.copy(), y_init.copy()


bo.fit(X_train, y_train)

selector = ModelSelectorMLE(
    bo=bo,
    X_train=X_train, 
    y_train=y_train,  
    kernel=bo.kernel,
    normalize_y=True, 
    normalize_X=True   
)

optimized_kernel = selector.find_via_scipy(
    method='L-BFGS-B',
    n_restarts=10,
    options={'maxiter': 500, 'ftol': 1e-6}
)

bo.gpr_.kernel = optimized_kernel
bo.kernel = optimized_kernel
bo.gpr_.optimizer = None
bo.fit(X_train, y_train)

logging.info(f"Final kernel after re-fit: {bo.gpr_.kernel}")

best_energy = y_train.min()
no_improvement_count = 0
results_dir = 'bo_test_results'


for iteration in range(N):
    if iteration % 10 == 0:
        X_train_scaled = bo.scaler_X.transform(X_train)
        y_train_scaled = bo.scaler_y.transform(y_train.reshape(-1, 1)).ravel()

        selector = ModelSelectorMLE(
            bo=bo,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            kernel=bo.kernel,
            normalize_y=False,
            normalize_X=False
        )

        new_kernel = selector.find_via_scipy(method='L-BFGS-B')
        logging.info(f"Re-optimized kernel: {new_kernel}")
        logging.info(f"New length_scale: {new_kernel.k1.length_scale:.4f}")

        bo.gpr_.kernel = new_kernel
        bo.kernel = new_kernel
        bo.gpr_.optimizer = None
        bo.fit(X_train, y_train)  # Refit GP

        logging.info(f"Updated kernel: {bo.gpr_.kernel}")

    logging.info(f"Iteration {iteration+1}/{N}")

    bo.fit(X_train, y_train)
    X_new = bo.suggest(batch_size=batch_size, xi=0.01, strategy=strategy)

    if len(X_new) == 0:
        logging.error("No suggested candidates. Check filters.")
        break

    for idx_conf, config_vector in enumerate(X_new):

        positions = config_vector.reshape(k, 3)

        atoms = atoms_host.copy()
        for pos in positions:
            atoms.append(Atom(int_atom, position=pos))

        atoms.wrap()

        calc_dir = Path(f"calc_iter{iteration+1:03d}_conf{idx_conf+1:03d}")
        calc_dir.mkdir(parents=True, exist_ok=True)

        atoms.calc = Vasp(directory=str(calc_dir), **vasp_params, kpts=(4, 6, 5))

        try:
            total_energy = atoms.get_potential_energy()
        except Exception as e:
            logging.error(f"Calculation for {calc_dir} failed: {e}")
            continue

        y_new = (total_energy - host_energy - k * int_atom_energy) / k

        X_train = np.vstack([X_train, config_vector.reshape(1, -1)])
        y_train = np.append(y_train, y_new)

        new_best_energy, improved = check_improvement(y_train, best_energy)
        if improved:
            logging.info(f"New best energy found: {new_best_energy:.6f} eV (previous: {best_energy:.6f} eV)")
            best_energy = new_best_energy
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            logging.info(f"No improvement in best energy for {no_improvement_count} iterations")


        new_row = pd.DataFrame([np.append(config_vector, y_new)],
                               columns=[f'x{i+1}' for i in range(3*k)] + ['energy'])
        header = not os.path.isfile('training_data.csv')
        new_row.to_csv('training_data.csv', mode='a', header=header, index=False)

        logging.info(f"Configuration {calc_dir} done: intercalation energy = {y_new:.6f} eV")


logging.info("Active learning done.") 



