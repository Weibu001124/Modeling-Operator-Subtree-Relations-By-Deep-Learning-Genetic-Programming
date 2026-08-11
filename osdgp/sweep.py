import os
import json
import sys
import time
import random
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

version = "Exp_"
folder = None
name = None
parallel = 10

def run_command(func, oplimits, popsize, decay_rate, seed, log_file):
    try:
        command = f'python3 main.py {func} {oplimits} {popsize} {decay_rate} {seed}'
        print(f'command: {command}')
        start_time = time.time()
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            stdout, stderr = proc.communicate(timeout=18000) # 5 hours
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

            os.makedirs(f"./{folder}/Err", exist_ok=True)
            err_path = f"./{folder}/Err/f{func}_run{seed}.log"
            with open(err_path, "w") as f:
                f.write(f"[TIMEOUT] Process exceeded 5 hours.\n")
                f.write(f"[STDOUT]\n{stdout}\n\n[STDERR]\n{stderr}\n")

            with open(log_file, 'a') as f:
                f.write(f"Run {seed}: Timeout after 5 hours.\n")

            return None, None, None, None, None, None, None, None, None, None, None, None, None, None
        
        # save standard output
        os.makedirs(f"./{version}/{folder}/Log", exist_ok=True)
        stdout_log = f"./{version}/{folder}/Log/log_pop{popsize}_run{seed}_{func}.txt"
        with open(stdout_log, "a") as f:
            f.write(stdout)

        if proc.returncode != 0:
            with open(log_file, 'a') as f:
                f.write(f"Run {seed}: Command failed with return code {proc.returncode}\n")
                f.write(f'Run {seed}: stderr:\n{stderr}\n')
                f.write(f'Run {seed}: stdout:\n{stdout}\n')
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None

        lines = stdout.splitlines()
        predy = trainmse = validmse = mae = mse = gen_val = cost_op = reduced_op_ratio = avg_cttimes_ratio = None
        model = model_size = model_simp = model_size_simp = memory = None
        for line in lines:
            # print(f'debug: {line}')
            if line.startswith("Predict Y"):
                try:
                    raw = line.split(":", 1)[1].strip()
                    raw = raw.replace('[', '').replace(']', '') # Remove brackets
                    nums = raw.split(',') # Split values by whitespace
                    predy = [float(v) for v in nums] # Convert to float list
                except Exception as e:
                    print(f"Warning: failed to parse Predict Y line. Reason: {e}")
                    predy = line.split(":", 1)[1].strip()   # or []   or keep original string
            elif line.startswith("Final MAE"):
                mae = float(line.split(":")[1].strip())
            elif line.startswith("Training MSE"):
                trainmse = float(line.split(":")[1].strip())
            elif line.startswith("Validation MSE"):
                validmse = float(line.split(":")[1].strip())
            elif line.startswith("Final MSE"):
                mse = float(line.split(":")[1].strip())
            elif line.startswith("Final R2"):
                rsquare = float(line.split(":")[1].strip())
            elif line.startswith("Total NFE"):
                nfe = int(line.split(":")[1].strip())
            elif line.startswith("Total OP"):
                cost_op = int(line.split(":")[1].strip())
            elif line.startswith("Total Gen"):
                gen_val = int(line.split(":")[1].strip())
            elif line.startswith("Best Program"):
                model = (line.split(":")[1].strip())
            elif line.startswith("Model Size"):
                model_size = int(line.split(":")[1].strip())
            elif line.startswith("Model Simplified"):
                model_simp = (line.split(":")[1].strip())
            elif line.startswith("Simplified Model Size"):
                model_size_simp = int(line.split(":")[1].strip())
            elif line.startswith("Max Memory Usage (GB)"):
                memory = float(line.split(":")[1].strip())
        end_time = time.time()
        with open(log_file, 'a') as f:
            status = "Succeed" if mse is not None and mse < 1e-18 else "Fail"
            f.write(f"Run {seed}: MAE={mae:.8f}, MSE={mse:.8f}, R2={rsquare:.8f}, Gen={gen_val}, NFE={nfe}, Cost OP={cost_op}, Model size={model_size}, Simplified Model size={model_size_simp}, Status={status}, Memory={memory:.8f}, Time={end_time - start_time:.2f}s\n")
        return predy, mae, trainmse, validmse, mse, rsquare, gen_val, nfe, cost_op, model, model_size, model_simp, model_size_simp, memory
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Run {seed}: Exception occurred - {e}\n")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

def run_repeat_pop(func, oplimits, pop, decay_rate, runtime):

    predys = []
    maes = []
    trainmses = []
    validmses = []
    mses = []
    r_squares = []
    cost_gens = []
    nfes = []
    cost_ops = []
    models = []
    modelsizes = []
    model_simps = []
    modelsize_simps = []
    memories = []

    os.makedirs(f"./{version}/{folder}/Results", exist_ok=True)
    log_file = f"./{version}/{folder}/Results/log_pop{pop}_{name}_f{func}.txt"
    with open(log_file, 'w') as f:
        f.write(f'Problem: f{func}\n')
        f.write(f'Population size: {pop}\n')

    futures = {}
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        for run_number in range(runtime):
            seed = random.randint(0, 99999) + run_number
            futures[executor.submit(run_command, func, oplimits, pop, decay_rate, seed, log_file)] = run_number

        for future in as_completed(futures):
            predy, mae, trainmse, validmse, mse, rsquare, gen_val, nfe, cost_op, model, modelsize, model_simp, model_size_simp, memory = future.result()
            if predy is not None:
                predys.append(predy)
            if trainmse is not None:
                trainmses.append(trainmse)
            if validmse is not None:
                validmses.append(validmse)
            if mse is not None:
                mses.append(mse)
            if mae is not None:
                maes.append(mae)
            if rsquare is not None:
                r_squares.append(rsquare)
            if gen_val is not None:
                cost_gens.append(gen_val)
            if nfe is not None:
                nfes.append(nfe)
            if cost_op is not None:
                cost_ops.append(cost_op)
            if model is not None:
                models.append(model)
            if modelsize is not None:
                modelsizes.append(modelsize)
            if model_simp is not None:
                model_simps.append(model_simp)
            if model_size_simp is not None:
                modelsize_simps.append(model_size_simp)
            if memory is not None:
                memories.append(memory)

    avg_mae = np.mean(maes) if maes else 0
    sstd_mae = np.std(maes, ddof=1) if len(maes) > 1 else 0
    median_mae = np.median(maes)

    avg_train_mse = np.mean(trainmses) if trainmses else 0
    sstd_train_mse = np.std(trainmses, ddof=1) if len(trainmses) > 1 else 0
    median_train_mse = np.median(trainmses)

    avg_valid_mse = np.mean(validmses) if validmses else 0
    sstd_valid_mse = np.std(validmses, ddof=1) if len(validmses) > 1 else 0
    median_valid_mse = np.median(validmses)

    avg_mse = np.mean(mses) if mses else 0
    sstd_mse = np.std(mses, ddof=1) if len(mses) > 1 else 0
    median_mse = np.median(mses)

    avg_r2 = np.mean(r_squares) if r_squares else 0
    std_r2 = np.std(r_squares, ddof=1) if len(r_squares) > 1 else 0

    avg_nfe = np.mean(nfes) if nfes else 0
    std_nfe = np.std(nfes, ddof=1) if len(nfes) > 1 else 0

    avg_op = np.mean(cost_ops) if cost_ops else 0
    std_op = np.std(cost_ops, ddof=1) if len(cost_ops) > 1 else 0

    avg_gen = np.mean(cost_gens) if cost_gens else 0
    std_gen = np.std(cost_gens, ddof=1) if len(cost_gens) > 1 else 0

    avg_model_size = np.mean(modelsizes) if modelsizes else 0
    std_model_size = np.std(modelsizes, ddof=1) if len(modelsizes) > 1 else 0

    avg_model_simp_size = np.mean(modelsize_simps) if modelsize_simps else 0
    std_model_simp_size = np.std(modelsize_simps, ddof=1) if len(modelsize_simps) > 1 else 0

    avg_mem = np.mean(memories) if memories else 0
    std_mem = np.std(memories, ddof=1) if len(memories) > 1 else 0

    with open(log_file, 'a') as f:
        f.write(f"Recorded Values (MAE): {maes}\n") # MAE
        f.write(f"Avg MAE: {avg_mae}\n")
        f.write(f"Sample Stdev MAE: {sstd_mae}\n")
        f.write(f"Median MAE: {median_mae}\n")
        f.write(f"Recorded Values (MSE): {mses}\n") # MSE
        f.write(f"Avg MSE: {avg_mse}\n")
        f.write(f"Sample Stdev MSE: {sstd_mse}\n")
        f.write(f"Median MSE: {median_mse}\n")
        f.write(f"Recorded Values (Validation MSE): {validmses}\n") # Validation MSE
        f.write(f"Avg Validation MSE: {avg_valid_mse}\n")
        f.write(f"Sample Stdev Validation MSE: {sstd_valid_mse}\n")
        f.write(f"Median Validation MSE: {median_valid_mse}\n")
        f.write(f"Recorded Values (R2): {r_squares}\n") # R Square
        f.write(f"Avg R2: {avg_r2}\n")
        f.write(f"Stdev R2: {std_r2}\n")
        f.write(f"Recorded Values (NFEs): {nfes}\n") # NFE
        f.write(f"Avg NFE: {avg_nfe}\n")
        f.write(f"Stdev NFE: {std_nfe}\n")
        f.write(f"Recorded Values (OPs): {cost_ops}\n") # OP
        f.write(f"Avg OP: {avg_op}\n")
        f.write(f"Stdev OP: {std_op}\n")
        f.write(f"Recorded Values (Model Size): {modelsizes}\n") # Model Sizes
        f.write(f"Avg Model Size: {avg_model_size}\n")
        f.write(f"Stdev Model Size: {std_model_size}\n")
        f.write(f"Recorded Values (Simplified Model Size): {modelsize_simps}\n") # Simplified Model Sizes
        f.write(f"Avg Simplified Model Size: {avg_model_simp_size}\n")
        f.write(f"Stdev Simplified Model Size: {std_model_simp_size}\n")
        f.write(f"Recorded Values (Max Memory Usage): {memories}\n") # Memory Usage
        f.write(f"Avg Max Memory Usage: {avg_mem}\n")
        f.write(f"Stdev Max Memory Usage: {std_mem}\n")

    return predys, maes, avg_mae, sstd_mae, median_mae, mses, avg_mse, sstd_mse, median_mse, trainmses, avg_train_mse, sstd_train_mse, median_train_mse, validmses, avg_valid_mse, sstd_valid_mse, median_valid_mse, r_squares, avg_r2, std_r2, cost_ops, avg_op, std_op, cost_gens, avg_gen, std_gen, models, modelsizes, avg_model_size, std_model_size, model_simps, modelsize_simps, avg_model_simp_size, std_model_simp_size, memories, avg_mem, std_mem

def sweep():
    if len(sys.argv) < 10:
        print("Usage: python sweep_const.py <date> <func_num> <oplimits> <decay_rate> <num_repeat> <parallel> <do_sweep> <folder> <algorithm>")
        return

    global folder, name, parallel
    date = sys.argv[1]
    func_num = int(sys.argv[2])
    oplimits = int(sys.argv[3])
    decay_rate = float(sys.argv[4])
    repeat = int(sys.argv[5])
    parallel = int(sys.argv[6])
    sweepmode = sys.argv[7]
    folder = sys.argv[8]
    name = sys.argv[9]

    global version
    version = version + date
    print(f'Folder: {version}')

    if sweepmode == "sweep":
        pops = [500, 400, 300, 200, 100, 50]
    elif sweepmode == "mixed":
        pops = [100, 1000]
    elif sweepmode == "1000":
        pops = [1000]
    elif sweepmode == "500":
        pops = [500]
    elif sweepmode == "100":
        pops = [100]

    print(f'Sweep: {pops}')
    print(f"Running sweep for f{func_num}")

    start_time = time.time()
    json_results = {}
    json_results = {
        "func_id": func_num
    }

    for pop in pops:
        print(f"--- Population {pop} ---")
        predys, maes, avg_mae, sstd_mae, median_mae, mses, avg_mse, sstd_mse, median_mse, trainmses, avg_train_mse, sstd_train_mse, median_train_mse, validmses, avg_valid_mse, sstd_valid_mse, median_valid_mse, r_squares, avg_r2, std_r2, cost_ops, avg_op, std_op, final_gens, avg_gen, std_gen, models, modelsizes, avg_model_size, std_model_size, model_simps, modelsize_simps, avg_model_simp_size, std_model_simp_size, memories, avg_mem, std_mem = run_repeat_pop(func_num, oplimits, pop, decay_rate, repeat)

        json_results[f"pop_{pop}"] = {
            "predict_y": predys, 
            "avg_mae": avg_mae, 
            "sstddev_mae": sstd_mae, 
            "median_mae": median_mae, 
            "maes": maes, 
            "avg_mse": avg_mse, 
            "stddev_mse": sstd_mse, 
            "median_mse": median_mse, 
            "mses": mses, 
            "avg_train_mse": avg_train_mse, 
            "stddev_train_mse": sstd_train_mse, 
            "median_train_mse": median_train_mse, 
            "trainmses": trainmses, 
            "avg_valid_mse": avg_valid_mse, 
            "stddev_valid_mse": sstd_valid_mse, 
            "median_valid_mse": median_valid_mse, 
            "validmses": validmses, 
            "avg_r2": avg_r2, 
            "stddev_r2": std_r2, 
            "rsquares": r_squares, 
            "avg_cost_op": avg_op, 
            "stddev_cost_op": std_op, 
            "cost_ops": cost_ops, 
            "avg_cost_gen": avg_gen, 
            "stddev_cost_gen": std_gen, 
            "final_gens": final_gens, 
            "models": models, 
            "avg_model_size": avg_model_size, 
            "stddev_model_size": std_model_size, 
            "model_sizes": modelsizes, 
            "models_simplified": model_simps, 
            "avg_simplified_model_size": avg_model_simp_size, 
            "stddev_simplified_model_size": std_model_simp_size, 
            "model_sizes_simplified": modelsize_simps, 
            "avg_memory": avg_mem, 
            "stddev_memory": std_mem, 
            "memories": memories
        }

    os.makedirs(f"./{version}/{folder}/Json", exist_ok=True)
    json_file = f"./{version}/{folder}/Json/log_f{func_num}.json"
    with open(json_file, "w") as jsonout:
        json.dump(json_results, jsonout, indent=2)

    print(f"Total time cost: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    sweep()

'''
python sweep.py date func_num oplimits decay_rate repeat parallel sweep_mode folder algo_record
python sweep.py 0811 1 10000000 0.8 2 2 1000 Test test
'''