import random as rd
import argparse
import json


def sample_patterns_list(number_atomic_propositions, pattern_length):
    # Generate a pattern of length pattern_length
    pattern = []
    for _ in range(pattern_length):
        state = ["?"] * number_atomic_propositions
        state[rd.randint(0, number_atomic_propositions - 1)] = 1
        pattern.append(state)
    return pattern


def generate_or_and_formula(min_index, max_index):
    # Generate a logical formula with size variables
    # min_index is included, max_index is excluded
    if min_index + 1 == max_index:
        return [min_index]
    else:
        op = rd.choice(["&&", "||"])
        cut = rd.randint(min_index + 1, max_index - 1)
        left_formula = generate_or_and_formula(min_index, cut)
        right_formula = generate_or_and_formula(cut, max_index)
        return [op, left_formula, right_formula]


def sample_or_and_formula_solution(or_and_formula):
    # Return a random solution of or_and_formula
    if or_and_formula[0] == "&&":
        left_solution = sample_or_and_formula_solution(or_and_formula[1])
        right_solution = sample_or_and_formula_solution(or_and_formula[2])
        return left_solution + right_solution
    elif or_and_formula[0] == "||":
        return sample_or_and_formula_solution(or_and_formula[rd.choice([1, 2])])
    else:
        return [or_and_formula[0]]


def is_or_and_formula_solution(or_and_formula, true_variables):
    # Verify that true_variables is solution of or_and_formula
    if or_and_formula[0] == "&&":
        left_bool = is_or_and_formula_solution(or_and_formula[1], true_variables)
        right_bool = is_or_and_formula_solution(or_and_formula[2], true_variables)
        return left_bool and right_bool
    elif or_and_formula[0] == "||":
        left_bool = is_or_and_formula_solution(or_and_formula[1], true_variables)
        right_bool = is_or_and_formula_solution(or_and_formula[2], true_variables)
        return left_bool or right_bool
    else:
        return or_and_formula[0] in true_variables


def generate_formula(number_atomic_propositions, number_patterns):
    # Generate a formula with number_patterns different patterns
    # The formula is a conjunction and disjunction of patterns
    patterns_list = []
    patterns_args = []
    arg_list = [2, 3]
    while number_patterns > 0:
        choice = rd.randint(0, 1)
        arg = arg_list[choice]
        pattern = sample_patterns_list(number_atomic_propositions, arg)
        if pattern not in patterns_list:
            patterns_list.append(pattern)
            patterns_args.append(arg)
            number_patterns -= 1
    or_and_formula = generate_or_and_formula(0, len(patterns_list))
    return patterns_list, patterns_args, or_and_formula


def generate_position_list(gap_list, max_length):
    # Return a list of positions not overlapping according to gap_list
    # max_length is excluded
    while True:
        position_list = rd.sample(range(max_length), len(gap_list))
        fits = True
        for i in range(len(position_list)):
            for j in range(1, gap_list[i]):
                # Checks that positions are spaced
                pos = position_list[i] + j
                if pos in position_list or pos >= max_length:
                    fits = False
        if fits:
            return position_list


def sample_positive_trace(
    number_atomic_propositions, trace_length, patterns_list, patterns_args, or_and_formula
):
    # Return a positive trace for the formula
    trace = [rd.choices([0, 1], k=number_atomic_propositions) for _ in range(trace_length)]
    or_and_formula_solution = sample_or_and_formula_solution(or_and_formula)
    filtered_patterns_list = [patterns_list[i] for i in or_and_formula_solution]
    filtered_patterns_args = [patterns_args[i] for i in or_and_formula_solution]
    position_list = generate_position_list(filtered_patterns_args, trace_length)
    # Modify trace to make it positive
    for pat, pos in zip(filtered_patterns_list, position_list):
        for i in range(len(pat)):
            for j in range(len(pat[i])):
                if pat[i][j] != "?":
                    trace[pos + i][j] = pat[i][j]
    return trace


def is_present_pattern(number_atomic_propositions, trace, pattern):
    # Check if pattern is in trace
    pos = 0
    while pos <= len(trace) - len(pattern):
        # Check that the trace satisfy the pattern on pos
        match = True
        ind = 0
        while match and ind < len(pattern):
            for i in range(number_atomic_propositions):
                if pattern[ind][i] == 1 and trace[pos + ind][i] == 0:
                    match = False
            ind += 1
        if match:
            return True
        pos += 1
    return False


def sample_negative_trace(number_atomic_propositions, trace_length, patterns_list, or_and_formula):
    # Return a negative trace for the formula
    lim = 10000
    while lim > 0:
        lim -= 1
        trace = [rd.choices([0, 1], k=number_atomic_propositions) for _ in range(trace_length)]
        present_patterns = []
        for i in range(len(patterns_list)):
            if is_present_pattern(number_atomic_propositions, trace, patterns_list[i]):
                present_patterns.append(i)
        if not is_or_and_formula_solution(or_and_formula, present_patterns):
            return trace
    return None


def pattern_to_formula(pattern):
    # Print the formula corresponding to pattern
    formula = "F"
    end = ""
    for state in pattern:
        for i, letter in enumerate(state):
            if letter == 1:
                formula += "(" + f"a{i}" + " && X[!]"
        end += ")"
    return formula[:-8] + end


def write_or_and_formula(or_and_formula, patterns_list):
    # Return a string corresponding to or_and_formula
    if len(or_and_formula) == 3:
        left_side = write_or_and_formula(or_and_formula[1], patterns_list)
        left_side = "(" + left_side + ")" if len(left_side) > 1 else left_side
        right_side = write_or_and_formula(or_and_formula[2], patterns_list)
        right_side = "(" + right_side + ")" if len(right_side) > 1 else right_side
        return left_side + " " + str(or_and_formula[0]) + " " + right_side
    else:
        return pattern_to_formula(patterns_list[or_and_formula[0]])


def main(
    trace_length,
    number_atomic_propositions,
    number_positive_traces,
    number_negative_traces,
    number_patterns,
    seed,
):
    rd.seed(seed)

    patterns_list, patterns_args, or_and_formula = generate_formula(
        number_atomic_propositions, number_patterns
    )
    positive_traces = set()
    while len(positive_traces) < number_positive_traces:
        positive_trace = sample_positive_trace(
            number_atomic_propositions, trace_length, patterns_list, patterns_args, or_and_formula
        )
        positive_trace_hashable = tuple(tuple(item) for item in positive_trace)
        positive_traces.add(positive_trace_hashable)

    negative_traces = set()
    while len(negative_traces) < number_negative_traces:
        negative_trace = sample_negative_trace(
            number_atomic_propositions, trace_length, patterns_list, or_and_formula
        )
        if negative_trace is None:
            raise Exception("failed positive trace generation")
        negative_trace_hashable = tuple(tuple(item) for item in negative_trace)
        negative_traces.add(negative_trace_hashable)

    formula = write_or_and_formula(or_and_formula, patterns_list)
    atomic_propositions = [f"a{i}" for i in range(number_atomic_propositions)]
    positive_traces = [
        dict(
            zip(
                atomic_propositions,
                [[x[i] for x in trace_tuple] for i in range(number_atomic_propositions)],
            )
        )
        for trace_tuple in positive_traces
    ]
    negative_traces = [
        dict(
            zip(
                atomic_propositions,
                [[x[i] for x in trace_tuple] for i in range(number_atomic_propositions)],
            )
        )
        for trace_tuple in negative_traces
    ]

    instance = {
        "positive_traces": positive_traces,
        "negative_traces": negative_traces,
        "smallest_known_formula": formula,
        "generating_formula": formula,
        "generating_seed": seed,
        "original_repository": "https://github.com/SynthesisLab/Bolt",
        "name": "RandomBooleanCombinationsofFactors",
        "parameters": {"number_patterns": number_patterns},
        "atomic_propositions": atomic_propositions,
        "number_atomic_propositions": number_atomic_propositions,
        "number_traces": number_positive_traces + number_negative_traces,
        "number_positive_traces": number_positive_traces,
        "number_negative_traces": number_negative_traces,
        "max_length_traces": trace_length,
        "trace_type": "finite",
    }

    param = (
        "trace_length="
        + str(trace_length)
        + "number_atomic_propositions="
        + str(number_atomic_propositions)
    )
    param = (
        param
        + "number_positive_traces="
        + str(number_positive_traces)
        + "number_negative_traces="
        + str(number_negative_traces)
        + "number_patterns="
        + str(number_patterns)
        + "seed="
        + str(seed)
    )
    with open("RandomBooleanCombinationsofFactors/" + param + ".json", "w") as f:
        json.dump(instance, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an LTL Learning instance for RandomBooleanCombinationsofFactors benchmark family."
    )
    parser.add_argument("-trace_length", type=int, default=10, help="Length of the traces.")
    parser.add_argument(
        "-number_atomic_propositions", type=int, default=3, help="Number of atomic propositions."
    )
    parser.add_argument(
        "-number_positive_traces", type=int, default=4, help="Number of positive traces."
    )
    parser.add_argument(
        "-number_negative_traces", type=int, default=4, help="Number of negative traces."
    )
    parser.add_argument("-number_patterns", type=int, default=4, help="Number of patterns.")
    parser.add_argument("-seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    main(
        args.trace_length,
        args.number_atomic_propositions,
        args.number_positive_traces,
        args.number_negative_traces,
        args.number_patterns,
        args.seed,
    )
