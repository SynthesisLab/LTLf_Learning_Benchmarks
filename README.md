# LTLf Learning Benchmarks

This repository contains 10 families of LTLf learning benchmarks. We include scripts to generate instances for all families, with many parameters allowing to generate instances with various number of traces and trace lengths. 

There are over 15_000 learning problems, which makes for 2GB of files, by only 100MB in zipped format so we included the zip file as releaset asset.

## Format

The benchmark files are structured in JSON format, with each file representing a learning problem. Each problem provides a set of positive and negative traces under the keys `positive_traces` and `negative_traces` respectively. Each trace is a dictionary where keys are atomic propositions (e.g., `"a0"`, `"a1"`) and values are lists of binary values representing the truth assignment of the proposition at each step in the trace. The `atomic_propositions` field explicitly lists all propositions used. Additional metadata such as `number_traces`, `number_positive_traces`, `number_negative_traces`, `max_length_traces`, and `trace_type` provide summary statistics about the trace sets. Fields like `smallest_known_formula`, `generating_formula`, `generating_seed`, `original_repository`, `name`, and `parameters` offer further information about the origin and characteristics of the benchmark problem.

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
        },
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

The main script is `sample.py`. It takes as an input a text file (such as [this one](./Fixed_Formulas/absence1.txt)) containing a formula. It creates an LTLf learning instance by sampling a given number of positive and negative traces of a given length uniformly at random. It builds upon the approach developed in [Scarlet](https://github.com/rajarshi008/Scarlet), compiling the LTL formula into a deterministic automaton and using it for sampling traces.


## Structure of this repository

* `Fixed_Formulas` includes:
	- a file `formulas.txt` where each line is a triple (formula, atomic_propositions, name). The list is extracted from [Flie](https://github.com/ivan-gavran/samples2LTL), which was inspired by [Property specification patterns for finite-state verification by Dwyer, Avrunin, and Corbett](https://dl.acm.org/doi/abs/10.1145/298595.298598).
	- instance files can be generated from the formulas in `formulas.txt` using `sample.py`. An instance is then called `fixed_formulas_trace_numbers=X_trace_length=Y.json`. We suggest the following parameters:
		* `X`: few = 5, much = 20, many = 100
		* `Y`: short = 16, medium = 32, long = 64

* `Generating_Instances` includes scripts named `gen_X.py` which take some `X`-specific parameters and return an instance in JSON format:
	- `X = Hamming` [VFB](https://github.com/MojtabaValizadeh/ltl-learning-on-gpus)
	- `X = OrderedSequence a0 U (a1 U (a2 ...))` [Syntcomp](https://github.com/SYNTCOMP/benchmarks/tree/master/tlsf-fin/Patterns/Uright)
	- `X = Subword F((a0 && XF((a1) && ...))` [Scarlet](https://github.com/rajarshi008/Scarlet)
	- `X = Subset F(a0) && F(a1) && F(a2) && ...` [Scarlet](https://github.com/rajarshi008/Scarlet), also [Syntcomp](https://github.com/SYNTCOMP/benchmarks/tree/master/tlsf-fin/Patterns/GFand)
	- `X = RandomBooleanCombinationsofFactors F(a0 && X(a1 && X(a2)))` (new in our paper)

Remark: we could first generate formulas using the scripts for these four families, and then use `sample.py` to generate traces. The trace generation for each of these four families is specific and therefore builds harder instances.

* `Generating_Formulas` includes scripts named `gen_X.py` which take some `X`-specific parameters and return a pair (formula, atomic_propositions):
	- `X = SingleCounter` [finite-synthesis](https://github.com/whitemech/finite-synthesis-datasets)
	- `X = DoubleCounter` [finite-synthesis](https://github.com/whitemech/finite-synthesis-datasets)
	- `X = Nim` [finite-synthesis](https://github.com/whitemech/finite-synthesis-datasets)
	- `X = RandomConjunctsFromBasis` reproducing [Symbolic LTLf Synthesis](https://arxiv.org/abs/1705.08426), inspired by [Improved Automata Generation for Linear Temporal Logic](https://link.springer.com/chapter/10.1007/3-540-48683-6_23)
