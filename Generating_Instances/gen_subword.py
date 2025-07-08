import random
import json
import argparse


def generate_subword(L: int, K: int, n_pos: int, n_neg: int, k: int, seed: int):
    """
    Generates an instance for LTL learning with specified parameters.

    Args:
        L: Length of the desired traces.
        K: Number of atomic propositions.
        n_pos: Number of positive traces to generate.
        n_neg: Number of negative traces to generate.
        k: Size of the desired subword.
        seed: Random seed.

    Returns:
        A JSON dictionary representing the LTL learning instance.
    """
    random.seed(seed)

    atomic_propositions = [f"a{i}" for i in range(K)]

    subword = random.choices(atomic_propositions, k=k)
    # print(subword)
    formula = write_subword_formula(subword)
    # print(formula)

    # Generate random positive traces
    positive_traces = []
    while len(positive_traces) < n_pos:
        positive_trace = generate_positive_trace(subword, k, L, atomic_propositions)
        # print(positive_trace)
        positive_traces.append(positive_trace)

    negative_traces_set = set()
    negative_traces = []
    while len(negative_traces) < n_neg:
        negative_trace = generate_negative_trace(subword, L, atomic_propositions)
        # check whether it contains the subword
        index = 0
        for i in range(L):
            if negative_trace[subword[index]][i] == 1:
                index += 1
            if index == k:
                break
        if index == k:
            continue

        negative_trace_hashable = tuple(tuple(negative_trace[atom]) for atom in atomic_propositions)

        if negative_trace_hashable not in negative_traces_set:
            negative_traces_set.add(negative_trace_hashable)
            negative_traces.append(negative_trace)

    instance = {
        "positive_traces": positive_traces,
        "negative_traces": negative_traces,
        "smallest_known_formula": formula,
        "generating_formula": formula,
        "generating_seed": seed,
        "original_repository": "https://github.com/rajarshi008/Scarlet",
        "name": "Subword",
        "parameters": {"subword_length": k},
        "atomic_propositions": atomic_propositions,
        "number_atomic_propositions": K,
        "number_traces": n_pos + n_neg,
        "number_positive_traces": n_pos,
        "number_negative_traces": n_neg,
        "max_length_traces": L,
        "trace_type": "finite",
    }

    param = "trace_length=" + str(L) + "number_atomic_propositions=" + str(K)
    param = (
        param
        + "number_positive_traces="
        + str(n_pos)
        + "number_negative_traces="
        + str(n_neg)
        + "subword_length="
        + str(k)
        + "seed="
        + str(seed)
    )
    with open("Subword/" + param + ".json", "w") as f:
        json.dump(instance, f)


def write_subword_formula(subword: list) -> dict:
    output = "F(" + subword[0] + " && "
    for a in subword[1:-1]:
        output = output + "X[!] F(" + a + " && "
    output = output + "X[!] F(" + subword[-1]
    output = output + ")" * len(subword)
    return output


def generate_positive_trace(subword: list, k: int, L: int, atomic_propositions: list) -> dict:
    positions = sorted(random.sample(range(L), k))
    # print(positions)
    pos2index = {x: i for i, x in enumerate(positions)}
    # print(pos2index)
    return {
        prop: [
            1 if i in positions and subword[pos2index[i]] == prop else random.choice([0, 1])
            for i in range(L)
        ]
        for prop in atomic_propositions
    }


def generate_negative_trace(subword: dict, L: int, atomic_propositions: list) -> dict:
    negative_trace = {
        prop: [random.choice([0, 1]) for _ in range(L)] for prop in atomic_propositions
    }
    return negative_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an LTL learning instance for subword benchmark family."
    )
    parser.add_argument("-trace_length", type=int, default=10, help="Length of the traces.")
    parser.add_argument(
        "-number_atomic_propositions", type=int, default=4, help="Number of atomic propositions."
    )
    parser.add_argument(
        "-number_positive_traces", type=int, default=4, help="Number of positive traces."
    )
    parser.add_argument(
        "-number_negative_traces", type=int, default=4, help="Number of negative traces."
    )
    parser.add_argument(
        "-subword_length", type=int, default=3, help="Length of the desired subword."
    )
    parser.add_argument("-seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    generate_subword(
        args.trace_length,
        args.number_atomic_propositions,
        args.number_positive_traces,
        args.number_negative_traces,
        args.subword_length,
        args.seed,
    )
