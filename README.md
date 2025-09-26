# $\mathsf{LTL}_f$-Learning Benchmarks

This repository contains $10$ families of $\mathsf{LTL}_f$-learning benchmarks.
We include scripts to generate instances for all families, with many parameters that allow generating instances with various numbers of traces and trace lengths.

There are over 15,000 $\mathsf{LTL}_f$-learning problems, which amount to 2 GB in total, but only about 100 MB in zipped format. We include the `.zip` file with all instances as a [release asset](https://github.com/SynthesisLab/LTLf_Learning_Benchmarks/releases/tag/benchmark).

## Format

The benchmark files are structured in JSON format, with each file representing a learning problem.
Each problem provides a set of positive and negative traces under the keys `positive_traces` and `negative_traces` respectively.
Each trace is a dictionary where keys are atomic propositions (e.g., `"a0"`, `"a1"`) and values are lists of binary values representing the truth assignment of the proposition at each step in the trace.
The `atomic_propositions` field explicitly lists all propositions used.
Additional metadata such as `number_traces`, `number_positive_traces`, `number_negative_traces`, `max_length_traces`, and `trace_type` provide summary statistics about the trace sets. Optional fields like `smallest_known_formula`, `generating_formula`, `generating_seed`, `original_repository`, `name`, and `parameters` offer further information about the origin and characteristics of the benchmark problem.

```json
{
    "positive_traces": [
        {
            "a0": [0, 1, 1, 0],
            "a1": [0, 1, 0, 1]
        }
    ],
    "negative_traces": [
        {
            "a0": [1, 1, 1, 0],
            "a1": [0, 1, 0, 1]
        },
        {
            "a0": [0, 1, 1, 0],
            "a1": [0, 1, 1, 1]
        },
        {
            "a0": [0, 1, 1, 1],
            "a1": [0, 1, 0, 1]
        },
        {
            "a0": [0, 1, 1, 0],
            "a1": [0, 1, 0, 0]
        }
    ],
    "atomic_propositions": ["a0", "a1"],
    "number_atomic_propositions": 2,
    "number_traces": 5,
    "number_positive_traces": 1,
    "number_negative_traces": 4,
    "max_length_traces": 4,
    "trace_type": "finite",
    "smallest_known_formula": "",
    "generating_formula": "",
    "generating_seed": "",
    "original_repository": "",
    "name": "",
    "parameters": {}
}
```

## Sampling

The main script is [`sample.py`](./sample.py).
It takes as an input a text file (such as [this one](./Fixed_Formulas/absence1.txt)) containing a semicolon-separated text file with four fields: `formula`; `atomic_propositions`;`name`;`source`.
It creates an $\mathsf{LTL}_f$-learning instance by sampling a given number of positive and negative traces of a given length uniformly at random.
It builds upon the approach developed in [Scarlet](https://github.com/rajarshi008/Scarlet), compiling the $\mathsf{LTL}_f$ formula into a deterministic automaton and using it to sample traces.

For instance, to generate a benchmark instance based on the formula file [absence1.txt](./Fixed_Formulas/absence1.txt) with $20$ positive traces and $20$ negative traces of length $16$, you can run

```python3 sample.py Fixed_Formulas/absence1.txt 20 16```

An optional fourth argument `--seed <seed>` allows to set a fixed seed for reproducibility (`42` by default).
The output is written to a file named `{file_name}_{nb_traces}_{length}_{seed}.json`.

The script [`sample.py`](./sample.py) uses the [Spot model-checker](https://spot.lre.epita.fr/) to compile $\mathsf{LTL}_f$ formulas into a deterministic automaton.
To use it, you need to install Spot and its Python bindings (as well as the Python bindings to the BDD library [BuDDy](https://buddy.sourceforge.net/manual/main.html), which is automatically installed along with Spot's Python bindings).

## Structure of this repository
The three folders in this repository all contain scripts to generate formulas and instances, combining various benchmarks from the literature in a unified format (as well as new benchmarks introduced in our paper).
We describe the content and source for each class of formulas/instances in these folders below.

* [`Fixed_Formulas`](./Fixed_Formulas):
	- The file [`formulas.txt`](./Fixed_Formulas/formulas.txt) contains multiple lines, each of which is a quadruple `formula`;`atomic_propositions`;`name`;`source`. The list is extracted from [Flie](https://github.com/ivan-gavran/samples2LTL), which was inspired by the article [*Property specification patterns for finite-state verification* by Dwyer, Avrunin, and Corbett](https://dl.acm.org/doi/abs/10.1145/298595.298598).
	- Instance files can then be generated from the formulas in this folder using `sample.py` directly. Our generated benchmarks include the following combinations of parameters:
		* `X` (the number of positive and negative traces): few = 5, much = 20, many = 100,
		* `Y` (the length of the traces): short = 16, medium = 32, long = 64.

      Instances in our benchmarks are then called `[name_of_fixed_formula]_X_Y_seed.json`.

* [`Generating_Instances`](./Generating_Instances) includes scripts named `gen_X.py` which take some `X`-specific parameters and return an instance in JSON format:
	- `X = Hamming`, which comes from the article [*LTL Learning on GPUs* by Valizadeh, Fijalkow, and Berger](https://link.springer.com/chapter/10.1007/978-3-031-65633-0_10) (see also the [GitHub](https://github.com/MojtabaValizadeh/ltl-learning-on-gpus));
	- `X = OrderedSequence`, generating formulas of the kind `a0 U (a1 U (a2 ...))`, which come from the Synthesis competition [SYNTCOMP](https://github.com/SYNTCOMP/benchmarks/tree/master/tlsf-fin/Patterns/Uright);
	- `X = Subword`, generating formulas of the kind `F(a0 && XF(a1 && ...))` which come from [Scarlet](https://github.com/rajarshi008/Scarlet);
	- `X = Subset`, generating formulas of the kind `F(a0) && F(a1) && F(a2) && ...`, which come from Scarlet and [SYNTCOMP](https://github.com/SYNTCOMP/benchmarks/tree/master/tlsf-fin/Patterns/GFand);
	- `X = RandomBooleanCombinationsofFactors`, generating formulas that are Boolean combinations of the pattern `F(a0 && X(a1 && X(a2)))`, which were introduced in our paper.

  **Remark**: we could also generate formulas using the scripts for these four families, and then use `sample.py` to generate traces. However, the trace generation script for each of these four families is specific to the family and therefore builds harder instances.

* [`Generating_Formulas`](./Generating_Formulas) includes scripts named `gen_X.py` which take some `X`-specific parameters and return a pair `(formula, atomic_propositions)`:
	- `X = SingleCounter`, `X = DoubleCounter`, and `X = Nim` all come from the [finite-synthesis-datasets](https://github.com/whitemech/finite-synthesis-datasets) repository, itself combining benchmarks from various sources,
	- `X = RandomConjunctsFromBasis` reproduces benchmarks from [Symbolic LTLf Synthesis](https://arxiv.org/abs/1705.08426), inspired by [Improved Automata Generation for Linear Temporal Logic](https://link.springer.com/chapter/10.1007/3-540-48683-6_23).
