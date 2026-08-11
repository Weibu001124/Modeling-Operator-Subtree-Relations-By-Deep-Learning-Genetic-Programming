# OSDGP

**Modeling Operator/Subtree Relations Using Deep Learning in Genetic Programming for Symbolic Regression**

OSDGP is a deep learning-guided genetic programming framework for symbolic regression. It models operator/subtree relations to guide donor subtree selection during crossover.

The final OSDGP configuration uses:

* **MLP with ReLU** for learning operator/subtree relations
* **Integer operator encoding**
* **Tree-LSTM with self-attention** for subtree embedding
* **Exponential decay** for weighting training instances across generations

## Requirements

* Python 3.10
* PyTorch
* NumPy
* SciPy
* scikit-learn
* Other dependencies specified in `environment.yml`

The Conda environment can be recreated with:

```bash
conda env create -f environment.yml
conda activate osdgp
```

## Usage

Run OSDGP with:

```bash
python3 main.py <problem_id> <NOP_limit> <population_size> <decay_rate> <random_state>
```

For example:

```bash
python3 main.py 1 10000000 1000 0.8 2026
```

The arguments are:

| Argument          | Description                         |
| ----------------- | ----------------------------------- |
| `problem_id`      | Identifier of the benchmark problem |
| `NOP_limit`       | Maximum number of operations        |
| `population_size` | Population size                     |
| `decay_rate`      | Exponential decay rate              |
| `random_state`    | Random seed                         |

## Repository Structure

```text
OSDGP/
├── Benchmark/          # Benchmark problems and datasets
├── src/                # Source code
├── main.py             # Main entry point
├── sweep.py            # Experiment configuration and parameter sweeps
├── environment.yml     # Conda environment
└── README.md
```

## Method Overview

During evolution, OSDGP learns the relations between parent operators and remaining subtrees from genetic operations.

For each crossover operation, the learned model takes the parent operator encoding and remaining subtree embedding as input and predicts a target donor subtree embedding. Candidate donor subtrees are then compared with the predicted embedding, and the candidate with the closest embedding is selected as the donor.

## Reproducibility

Experiments use specified random seeds to support reproducibility. The random seed can be provided through the `random_state` argument.

## Citation

If you use this code in your research, please cite the corresponding OSDGP paper.