import random
import json
import argparse

def generate_Hamming(L: int, K: int, n_neg: int, delta: int, seed: int) -> dict:
    """
    Generates an instance for LTLf learning with specified parameters.

    Args:
        L: Length of the desired traces.
        K: Number of atomic propositions.
        n_neg: Number of negative traces to generate.
        delta: Maximum Hamming distance between positive and negative traces.
        seed: Random seed.

    Returns:
        A JSON dictionary representing the LTLf learning instance.
    """
    random.seed(seed)

    atomic_propositions = [f"a{i}" for i in range(K)]

    # Generate a single random positive trace
    positive_trace = {prop: [random.choice([0, 1]) for _ in range(L)] for prop in atomic_propositions}
    positive_traces = [positive_trace]

    negative_traces = set()
    while len(negative_traces) < n_neg:
        negative_trace = generate_negative_trace(positive_trace, delta)
        negative_trace_hashable = tuple(tuple(tuple(item[1]) for item in negative_trace.items()))
        negative_traces.add(negative_trace_hashable)

    # Convert back to the desired dictionary format
    formatted_negative_traces = [dict(zip(atomic_propositions, [list(x) for x in trace_tuple])) for trace_tuple in negative_traces]

    instance = {
        "positive_traces": positive_traces,
        "negative_traces": formatted_negative_traces,
        "smallest_known_formula": "",
        "generating_formula": "",
        "generating_seed": seed,
        "original_repository": "https://github.com/MojtabaValizadeh/ltl-learning-on-gpus",
        "name": "Hamming",
        "parameters": {"delta": delta},
        "atomic_propositions": atomic_propositions,
        "number_atomic_propositions": K,
        "number_traces": 1 + n_neg,
        "number_positive_traces": 1,
        "number_negative_traces": n_neg,
        "max_length_traces": L,
        "trace_type": "finite",
    }
    param = "trace_length=" + str(L) + "number_atomic_propositions=" + str(K) + "number_negative_traces=" + str(n_neg) + "delta=" + str(delta) + "seed=" + str(seed)
    with open("Hamming/" + param + ".json", "w") as f:
        json.dump(instance, f)

def generate_negative_trace(positive_trace: dict, delta: int) -> dict:
    """
    Generates a negative trace at a Hamming distance of at most delta from the positive trace.

    Args:
        positive_trace: The positive trace dictionary.
        delta: The maximum Hamming distance.

    Returns:
        A dictionary representing a negative trace.
    """
    negative_trace = {prop: list(values) for prop, values in positive_trace.items()} # Create a mutable copy
    length = len(list(positive_trace.values())[0])
    num_flips = random.randint(1, delta) # Ensure at least one flip for negativity
    indices_to_flip = random.sample(range(length), num_flips)
    prop_to_flip = random.choice(list(positive_trace.keys())) # Choose one proposition to modify

    for index in indices_to_flip:
        negative_trace[prop_to_flip][index] = 1 if negative_trace[prop_to_flip][index] == 0 else 0

    return negative_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an LTL learning instance for Hamming distance benchmark family.")
    parser.add_argument("-trace_length", type=int, default=10, help="Length of the traces.")
    parser.add_argument("-number_atomic_propositions", type=int, default=3, help="Number of atomic propositions.")
    parser.add_argument("-number_negative_traces", type=int, default=4, help="Number of negative traces.")
    parser.add_argument("-delta", type=int, default=1, help="Maximum Hamming distance between positive and negative traces.")
    parser.add_argument("-seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    generate_Hamming(args.trace_length, args.number_atomic_propositions, args.number_negative_traces, args.delta, args.seed)
