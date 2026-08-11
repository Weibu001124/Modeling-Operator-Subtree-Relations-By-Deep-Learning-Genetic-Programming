import sys
import math
import random
import math
import time
import resource
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.gplearn.genetic import SymbolicRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    if len(sys.argv) < 6:
        print("Usage: python main.py <func> <op> <pop> <decay_rate> <seed>")
        sys.exit(1)

    start = time.time()
    func = int(sys.argv[1])
    op = int(sys.argv[2])
    pop_size = int(sys.argv[3])
    decay_rate = float(sys.argv[4])
    seed = int(sys.argv[5])

    print(f'Problem: f{func}')
    func_to_files = {
        1: ("./Benchmark/koza3_train.txt", "./Benchmark/koza3_test.txt"), 
    }

    if func in func_to_files:
        training_data, testing_data = func_to_files[func]
    else:
        print(f"Function {func} not found.")
        return

    f1 = open(training_data)
    X_train = []
    y_train = []
    for line in f1.readlines():
        nums = list(map(float, line.strip().split()))
        X_train.append(nums[:-1])
        y_train.append(nums[-1])
    f1.close()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.ravel()

    f2 = open(testing_data)
    X_test = []
    y_test = []
    for line in f2.readlines():
        nums = list(map(float, line.strip().split()))
        X_test.append(nums[:-1])
        y_test.append(nums[-1])
    f2.close()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"Loaded data: Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
    print(f"========================================= Random State {seed} =========================================")
    hyperparams = [{}]
    est_gp = SymbolicRegressor(
            validation=1/3,
            mode=1, # 0: gplearn, 1: osdgp
            decay_rate=decay_rate, 
            population_size=pop_size,
            generations=10000,
            opevals=op,
            tournament_size=20,
            stopping_criteria=1e-18,
            const_range=(-1., 1.), 
            init_depth=(2, 6),
            init_method='half and half', 
            function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp'),
            metric='mse', # mean absolute error
            parsimony_coefficient=0.001, 
            p_crossover=0.9,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0,
            p_point_mutation=0,
            p_point_replace=0,
            max_samples=1.0,
            n_jobs=1,
            verbose=1,
            random_state=seed
    )

    est_gp.fit(X_train, y_train, None)
    end = time.time()
    max_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem_gb = max_mem_kb / (1024 * 1024)
    max_mem_str = f'{max_mem_kb / (1024 * 1024):.2f} GB'
    
    model = est_gp._program
    model_size = est_gp._program._length()
    print('Best Program:', model)
    print('Model Size:', model_size)

    y_pred = est_gp.predict(X_test)
    print('Training MSE:', est_gp.get_trainfit())
    print('Validation MSE:', est_gp.get_validfit())
    print('Testing MSE:', mean_squared_error(y_test, y_pred))
    print('Final MAE:', mean_absolute_error(y_test, y_pred))
    print('Final MSE:', mean_squared_error(y_test, y_pred))
    print('Final R2:', r2_score(y_test, y_pred))
    print('Total OP:', est_gp.get_opevals())
    print('Total NFE:', est_gp.get_evaluations())
    print('Total Gen:', est_gp.get_generations())

    '''
    ############ Simplification ############
    import sympy
    from sympy import Function, symbols
    import signal
    from utils import simplify_gplearn_program, count_nodes_as_gplearn

    def timeout_handler(signum, frame):
        raise TimeoutError("Sympy simplification exceeded time limit")

    # set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300) # 5 mins

    try:
        print("Simplifying with Sympy...")
        try:
            simplified_expr = simplify_gplearn_program(model)
            simplified_model_size = count_nodes_as_gplearn(simplified_expr)
        finally:
            signal.alarm(0)   # always disable alarm
    except Exception as e:
        print(f"Sympy failed — using original model. Reason: {type(e).__name__}: {e}")
        simplified_expr = model
        simplified_model_size = model_size
    finally:
        signal.alarm(0)

    model = str(simplified_expr)
    print('Model Simplified:', model)
    print('Simplified Model Size:', simplified_model_size)
    ############ ############# ############
    '''

    print(f'Execution time: {end - start} secs')
    print(f'Max Memory Usage (GB):', max_mem_gb)

if __name__ == '__main__':
    main()

'''
run this
python3 main.py function op population decayrate run
python3 main.py 1 10000000 1000 0.8 2026
'''