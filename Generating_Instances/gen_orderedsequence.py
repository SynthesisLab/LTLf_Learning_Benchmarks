import random
import json
import argparse

def generate_orderedsequence(L: int, K: int, k: int, n_pos: int, n_neg: int, seed: int):
    """
    Generates an instance for LTL learning with specified parameters.

    Args:
        L: Length of the desired traces.
        K: Number of atomic propositions.
        k: Size of the desired ordered sequence.
        n_pos: Number of positive traces to generate.
        n_neg: Number of negative traces to generate.
        seed: Random seed.

    Returns:
        A JSON dictionary representing the LTL learning instance.
    """
    random.seed(seed)

    atomic_propositions = [f"a{i}" for i in range(K)]

    orderedsequence = []
    curr = None
    i = 0
    while i < k:
        nxt = random.choice(atomic_propositions)
        if nxt != curr:
            orderedsequence.append(nxt)
            curr = nxt
            i += 1
    # print(orderedsequence)
    formula = write_orderedsequence_formula(orderedsequence)
    # print(formula)

    # Generate random positive traces
    positive_traces = []
    while len(positive_traces) < n_pos:
        positive_trace = generate_positive_trace(orderedsequence, k, L, atomic_propositions)
        # print(positive_trace)
        positive_traces.append(positive_trace)

    negative_traces = set()
    while len(negative_traces) < n_neg:
        negative_trace = generate_negative_trace(orderedsequence, L, atomic_propositions)
        # check whether it contains the orderedsequence
        # start from the back and see if we can reach 0
        # print(negative_trace)
        sat = {i: negative_trace[orderedsequence[k-1]][i] for i in range(L)}
        for curr in range(k-2,-1,-1):
            # print(curr, sat)
            next_one_in_sat = {i: -1 for i in range(L)}
            index_next_one = L+1
            for j in range(L-1,-1,-1):
                if sat[j] == 1:
                    index_next_one = j
                next_one_in_sat[j] = index_next_one

            prop = orderedsequence[curr]
            next_zero = {i: i for i in range(L)}
            index_next_zero = L
            for j in range(L-1,-1,-1):
                if negative_trace[prop][j] == 0:
                    index_next_zero = j
                next_zero[j] = index_next_zero
            new_sat = {i: 0 for i in range(L)}
            for i in range(L-1,-1,-1):
                if next_zero[i] >= next_one_in_sat[i]:
                    new_sat[i] = 1 
            sat = new_sat
        if sat[0] == 0:
            # print("is indeed negative")
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
        "name": "OrderedSequence",
        "parameters": {"size_sequence": k},
        "atomic_propositions": atomic_propositions,
        "number_atomic_propositions": K,
        "number_traces": n_pos + n_neg,
        "number_positive_traces": n_pos,
        "number_negative_traces": n_neg,
        "max_length_traces": L,
        "trace_type": "finite",
    }

    param = "trace_length=" + str(L) + "number_atomic_propositions=" + str(K) + "size_sequence=" + str(k) + "number_positive_traces=" + str(n_pos) + "number_negative_traces=" + str(n_neg) + "seed=" + str(seed)
    with open("OrderedSequence/" + param + ".json", "w") as f:
        json.dump(instance, f)

def write_orderedsequence_formula(orderedsequence: list) -> dict:
    output = ""    
    for a in orderedsequence[:-1]:
        output = output + a + " U ("
    output = output + orderedsequence[-1]
    output = output + ")" * len(orderedsequence)
    return output

def generate_positive_trace(orderedsequence: list, k: int, L: int, atomic_propositions: list) -> dict:
    positions = sorted(random.sample(range(L), k-1))
    # print(positions)
    fixed = [""] * L 
    curr = 0
    for i in range(L):
        if curr == k-1:
            continue
        if i == positions[curr] and curr < k-1:
            curr += 1
        fixed[i] = orderedsequence[curr]
    # print(fixed)
    return {prop: [1 if fixed[i] == prop else random.choice([0, 1]) for i in range(L)] for prop in atomic_propositions}

def generate_negative_trace(orderedsequence: dict, L: int, atomic_propositions: list) -> dict:
    negative_trace = {prop: [random.choice([0, 1]) for _ in range(L)] for prop in atomic_propositions}
    return negative_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an LTL Learning instance for orderedsequence benchmark family.")
    parser.add_argument("-trace_length", type=int, default=10, help="Length of the traces.")
    parser.add_argument("-number_atomic_propositions", type=int, default=3, help="Number of atomic propositions.")
    parser.add_argument("-size_sequence", type=int, default=3, help="Size of the desired orderedsequence.")
    parser.add_argument("-number_positive_traces", type=int, default=4, help="Number of positive traces.")
    parser.add_argument("-number_negative_traces", type=int, default=4, help="Number of negative traces.")
    parser.add_argument("-seed", type=int, default=45, help="Random seed.")

    args = parser.parse_args()

    generate_orderedsequence(args.trace_length, args.number_atomic_propositions, args.size_sequence, args.number_positive_traces, args.number_negative_traces, args.seed)
