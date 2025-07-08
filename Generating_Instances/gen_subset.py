import random
import json
import argparse

def generate_subset(L: int, K: int, n_pos: int, n_neg: int, k: int, seed: int):
    """
    Generates an instance for LTL learning with specified parameters.

    Args:
        L: Length of the desired traces.
        K: Number of atomic propositions.
        n_pos: Number of positive traces to generate.
        n_neg: Number of negative traces to generate.
        k: Size of the desired subset.
        seed: Random seed.

    Returns:
        A JSON dictionary representing the LTL learning instance.
    """
    random.seed(seed)

    atomic_propositions = [f"a{i}" for i in range(K)]

    subset = random.sample(atomic_propositions, k=k)
    # print(subset)
    formula = write_subset_formula(subset)
    # print(formula)

    # Generate random positive traces
    positive_traces = []
    while len(positive_traces) < n_pos:
        positive_trace = generate_positive_trace(subset, k, L, atomic_propositions)
        # print(positive_trace)
        positive_traces.append(positive_trace)

    negative_traces = set()
    while len(negative_traces) < n_neg:
        negative_trace = generate_negative_trace(subset, k, L, atomic_propositions)
        negative_trace_hashable = tuple(tuple(tuple(item[1]) for item in negative_trace.items()))
        negative_traces.add(negative_trace_hashable)

    # Convert back to the desired dictionary format
    formatted_negative_traces = [dict(zip(atomic_propositions, [list(x) for x in trace_tuple])) for trace_tuple in negative_traces]

    instance = {
        "positive_traces": positive_traces,
        "negative_traces": formatted_negative_traces,
        "smallest_known_formula": formula,
        "generating_formula": formula,
        "generating_seed": seed,
        "original_repository": "https://github.com/rajarshi008/Scarlet",
        "name": "Subset",
        "parameters": {"subset_size": k},
        "atomic_propositions": atomic_propositions,
        "number_atomic_propositions": K,
        "number_traces": n_pos + n_neg,
        "number_positive_traces": n_pos,
        "number_negative_traces": n_neg,
        "max_length_traces": L,
        "trace_type": "finite",
    }

    param = "trace_length=" + str(L) + "number_atomic_propositions=" + str(K)
    param = param + "number_positive_traces=" + str(n_pos) + "number_negative_traces=" + str(n_neg) + "subset_size=" + str(k) + "seed=" + str(seed)
    with open("Subset/" + param + ".json", "w") as f:
        json.dump(instance, f)

def write_subset_formula(subset: list) -> dict:
    output = ""
    for a in subset[:-1]:
        output = output + "F(" + a + ") && "
    output = output + "F(" + subset[-1] + ")"
    return output

def generate_positive_trace(subset: list, k: int, L: int, atomic_propositions: list) -> dict:
    positions = random.sample(range(L), k)
    # print(positions)
    pos2index = {x: i for i, x in enumerate(positions)}
    # print(pos2index)
    return {prop: [1 if i in positions and subset[pos2index[i]] == prop else random.choice([0, 1]) for i in range(L)] for prop in atomic_propositions}

def generate_negative_trace(subset: list, k: int, L: int, atomic_propositions: list) -> dict:
    forbidden_prop = random.choice(subset)
    negative_trace = {prop: [0] * L if prop == forbidden_prop else [random.choice([0, 1]) for _ in range(L)] for prop in atomic_propositions}
    return negative_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an LTL Learning instance for subset benchmark family.")
    parser.add_argument("-trace_length", type=int, default=10, help="Length of the traces.")
    parser.add_argument("-number_atomic_propositions", type=int, default=4, help="Number of atomic propositions.")
    parser.add_argument("-number_positive_traces", type=int, default=4, help="Number of positive traces.")
    parser.add_argument("-number_negative_traces", type=int, default=4, help="Number of negative traces.")
    parser.add_argument("-subset_size", type=int, default=3, help="Size of the desired subset.")
    parser.add_argument("-seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    generate_subset(args.trace_length, 
        args.number_atomic_propositions, 
        args.number_positive_traces, 
        args.number_negative_traces, 
        args.subset_size, 
        args.seed)