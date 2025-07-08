from dataclasses import dataclass
import random
import spot
import buddy
import logging


class Trace:
    """
    Class to store a trace.
    """

    trace: list[list[int]]  # trace[i] = i-th letter of the trace, i.e., a list of

    # 0s and 1s where the j-th element is the value of
    # the j-th propositional variable in the same order
    # as in `ap`.
    def __init__(self, ap: list[str], length: int) -> None:
        self.trace = [[0] * len(ap) for _ in range(length)]

    def format_json(self, ap) -> dict[list[int]]:
        """
        Returns the trace as a list of lists of 0s and 1s, where list[i][j]
        is the value of the i-th atomic proposition (from ap) at the j-th
        position in the trace.

        Note that this is the transpose of the way the trace is stored in
        the class.
        """
        assert len(self.trace[0]) == len(ap), "Invalid trace length: {} != {}".format(
            len(self.trace[0]), len(ap)
        )
        return {ap[i]: [self.trace[j][i] for j in range(len(self.trace))] for i in range(len(ap))}

    def __str__(self) -> str:
        return str(self.trace)

    def __repr__(self) -> str:
        return str(self.trace)

    def __len__(self) -> int:
        return len(self.trace)

    def __getitem__(self, i: int) -> list[int]:
        return self.trace[i]

    def __setitem__(self, i: int, letter: list[int]) -> None:
        self.trace[i] = letter

    def __eq__(self, value) -> bool:
        if isinstance(value, Trace):
            return len(self.trace) == len(value.trace) and all(
                len(self.trace[i]) == len(value.trace[i])
                and all(self.trace[i][j] == value.trace[i][j] for j in range(len(self.trace[i])))
                for i in range(len(self.trace))
            )
        return False

    def __lt__(self, value) -> bool:
        return self.trace < value.trace

    def __hash__(self):
        return hash(tuple(tuple(letter) for letter in self.trace))


def _is_terminal(bdd: buddy.bdd) -> bool:
    """
    Returns True if `bdd` is a terminal node (i.e., true or false).
    """
    return bdd == buddy.bddtrue or bdd == buddy.bddfalse


def _variable_ordering(bdict: spot.impl.bdd_dict, ap: list[str]) -> list[int]:
    """
    Returns a variable ordering consistent with the BDD (in practice, an array containing the `varnum` of each variable in BDD order).

    The code relies on the property that the BDD order is consistent with
    the variable number (`varnum`), which is true in Spot when the BDD order
    is not modified.
    """
    var_num_pairs = [(a, bdict.varnum(a)) for a in ap]
    var_num_pairs.sort(key=lambda x: x[1])
    return [x[1] for x in var_num_pairs]


class Ordering:
    """
    Class to store the ordering of variables in a BDD, and then obtain
    various information about the ordering efficiently.
    """

    _var_order: list  # index in BDD order -> varnum
    _var_to_position: dict  # varnum -> index in BDD order
    _var_to_ap_index: dict  # varnum -> index in alphabet order

    def __init__(self, bdict: spot.impl.bdd_dict, ap: list[str]) -> None:
        self._var_order = _variable_ordering(bdict, ap)
        self._var_to_position = {var: i for i, var in enumerate(self._var_order)}
        self._var_to_ap_index = {bdict.varnum(a): i for i, a in enumerate(ap)}

    def position2var(self, position: int) -> int:
        """
        Returns the varnum of the variable at the given `position` in the
        BDD order.
        """
        return self._var_order[position]

    def var2position(self, var: int) -> int:
        """
        Returns the index of the variable in the BDD order from its varnum
        `var`.
        """
        return self._var_to_position[var]

    def var2apindex(self, var: int) -> int:
        """
        Returns the index of the variable in alphabet order (i.e., from
        ap) from its varnum `var`.
        """
        return self._var_to_ap_index[var]

    def position2apindex(self, position: int) -> int:
        """
        Returns the index of the variable in alphabet order (i.e., from
        ap) from its index `position` in BDD order.
        """
        return self._var_to_ap_index[self._var_order[position]]

    def len(self) -> int:
        """
        Returns the number of variables in the ordering.
        """
        return len(self._var_order)


@dataclass
class AutomatonCache:
    """
    Class to store memoized information about an automaton.
    """

    formula: str
    ap: list[str]
    automaton: spot.twa_graph
    length: int
    order: Ordering
    counts: dict[buddy.bdd, int]
    counts_true: dict[buddy.bdd, int]
    prog_dyn_table: dict[int, dict[int, int]]
    prog_dyn_table_transitions: dict[
        int, dict[int, list[tuple[int, spot.impl.twa_graph_edge_storage]]]
    ]


def count_satisfying_assignments(bdd: buddy.bdd, cache: AutomatonCache) -> int:
    """
    Returns the number of satisfying assignments of a BDD.

    Updates the optional `counts` and `counts_true` dictionaries with the
    number of satisfying assignments for each node in the BDD descendent
    from `bdd`. The dictionaries only take into account the variables
    occurring after the variable of `bdd` in the BDD order for counting.
    This allows to memoize the results and avoid recomputing them at each
    call to the same shared BDD.
    """
    order = cache.order

    def rec_count(node):
        if node in cache.counts:
            # Means that the node and its descendents are already counted
            return cache.counts[node]

        if _is_terminal(node):
            # Means that the node is a true or false node with no descendent
            count = 1 if node == buddy.bddtrue else 0
            cache.counts[node] = count
            return count

        var = buddy.bdd_var(node)
        current_var_index = order.var2position(var)
        true_node = buddy.bdd_high(node)
        false_node = buddy.bdd_low(node)
        true_count = rec_count(true_node)
        false_count = rec_count(false_node)
        true_var_index = (
            order.len() if _is_terminal(true_node) else order.var2position(buddy.bdd_var(true_node))
        )
        false_var_index = (
            order.len()
            if _is_terminal(false_node)
            else order.var2position(buddy.bdd_var(false_node))
        )

        # Check that the variable ordering is consistent with the BDD order
        assert true_var_index > current_var_index, (
            "Variable ordering problem: true_var_index: {}, current_var_index: {}".format(
                true_var_index, current_var_index
            )
        )
        assert false_var_index > current_var_index, (
            "Variable ordering problem: false_var_index: {}, current_var_index: {}".format(
                false_var_index, current_var_index
            )
        )

        # Counts for the true branch of a node; no need to count for the false
        # branch, as we can compute it from counts - counts_true.
        # Note that counts_true[node] may differ from counts[node.high] because
        # there may be variables in the BDD order that do not appear along this
        # branch in the BDD. We should therefore multiply the counts by
        # 2^(number of variables in the ordering not appearing along this
        # branch).
        cache.counts_true[node] = (2 ** (true_var_index - current_var_index - 1)) * true_count
        count = (
            cache.counts_true[node] + (2 ** (false_var_index - current_var_index - 1)) * false_count
        )
        cache.counts[node] = count
        return count

    # Multiply by 2^(number of variables in the BDD order before the root
    # variable)
    first_var_index = order.len() if _is_terminal(bdd) else order.var2position(buddy.bdd_var(bdd))

    return (2 ** (first_var_index)) * rec_count(bdd)


def sample_letter_from_BDD(bdd: buddy.bdd, cache: AutomatonCache) -> list[int]:
    """
    Samples a satisfying assignment from a BDD uniformly at random.
    Returns it as a list of 0s and 1s, where the i-th element is the value
    of the i-th propositional variable in the same order as in `ap`.
    Note that the order of the variables in `ap` may be different from the
    order of the variables in the BDD.

    `var_order` is an optional argument that allows to specify the BDD order
    if already computed. This is useful to avoid recomputing the order
    every time we sample from the same BDD.
    `counts` and `counts_true` are optional dictionaries that allow to
    memoize the counts of the BDD. This is useful to avoid recomputing
    the counts every time we sample from the same BDD.
    """
    # First, count the number of satisfying assignments from each node
    # and store the results in counts and counts_true.
    count_satisfying_assignments(bdd, cache)

    # We will now generate a random assignment
    assignment = [-1] * len(cache.ap)

    # Helper function to assign random values to the variables between indices
    # start and end - 1 in the BDD order. The variables are in the order of
    # var_order, and the values are assigned to the corresponding
    # variables in the order of ap.
    def assign_randoms(start, end):
        for i in range(start, end):
            assignment[cache.order.position2apindex(i)] = random.randint(0, 1)

    node = bdd
    previous_var_index = -1
    while not _is_terminal(node):
        var = buddy.bdd_var(node)
        next_var_index = cache.order.var2position(var)
        assign_randoms(previous_var_index + 1, next_var_index)

        prob_true = cache.counts_true[node] / cache.counts[node]
        if random.random() < prob_true:
            assignment[cache.order.var2apindex(var)] = 1
            node = buddy.bdd_high(node)
        else:
            assignment[cache.order.var2apindex(var)] = 0
            node = buddy.bdd_low(node)
        previous_var_index = next_var_index

    # Assign random values to the remaining variables in the ordering
    assign_randoms(previous_var_index + 1, len(cache.ap))

    # Check that all variables have been assigned to 0 or 1
    assert all([x == 0 or x == 1 for x in assignment]), "The assignment is not valid: {}".format(
        assignment
    )

    return assignment


def compile(formula: str, ap: list[str]) -> spot.twa_graph:
    f = spot.from_ltlf(formula)
    aut = f.translate("buchi", "sbacc", "deterministic")
    automaton = spot.to_finite(aut)
    # All atomic propositions may not be used in the formula, so we need to
    # register them in the automaton
    if ap is not None:
        for a in ap:
            automaton.register_ap(a)
    spot.complete_here(automaton)  # Completeness is used when sampling traces
    # outside the language, as it ensures that
    # non-accepted traces can be read
    return automaton


def _prog_dyn_table_length(cache: AutomatonCache) -> None:
    """
    Updates the cache with two dictionaries (for all 'q' a state of
    `automaton` and	'l' <= `max_length`):
    - prog_dyn_table[l][q] = number of traces of length 'l' reaching state 'q' from initial state
    - prog_dyn_table_transitions[l][q] = list of pairs (x,y) where
    x = number of traces of length 'l' reaching state 'q' from initial state with last incoming transition y.
    """
    automaton = cache.automaton
    length = cache.length

    # Initialize the dynamic programming tables
    prog_dyn_table = {
        length: {q: 0 for q in range(automaton.num_states())} for length in range(length + 1)
    }
    prog_dyn_table_transitions = {
        length: {state: [] for state in range(automaton.num_states())}
        for length in range(length + 1)
    }

    # Base case: empty trace
    prog_dyn_table[0][automaton.get_init_state_number()] = 1
    # Compute the number of traces iteratively
    for l in range(length):
        for q in range(automaton.num_states()):
            if prog_dyn_table[l][q] > 0:
                for transition in automaton.out(q):
                    qn = transition.dst
                    formula = transition.cond
                    number_letters = count_satisfying_assignments(formula, cache)
                    number_traces = number_letters * prog_dyn_table[l][q]
                    prog_dyn_table[l + 1][qn] += number_traces
                    prog_dyn_table_transitions[l + 1][qn].append((number_traces, transition))

    # The random library struggles with large integers, so we limit them.
    MAX_WEIGHT = 2**1023
    for l in range(length + 1):
        total_weights_l = sum(prog_dyn_table[l].values())
        if total_weights_l >= MAX_WEIGHT:
            # Reduce the weights to avoid float overflow
            # This may introduce a very small bias in the uniform sampling
            divisor = total_weights_l // MAX_WEIGHT
            assert divisor > 0, "Invalid divisor: {}".format(divisor)
            logging.info("Huge weights in the automaton: dividing weights by {}".format(divisor))

            def reduce(x):
                if x > 0:
                    x //= divisor
                    x = max(x, 1)
                return x

            prog_dyn_table[l] = {
                state: reduce(prog_dyn_table[l][state]) for state in prog_dyn_table[l]
            }
            for q in range(automaton.num_states()):
                prog_dyn_table_transitions[l][q] = [
                    (reduce(x), y) for (x, y) in prog_dyn_table_transitions[l][q]
                ]

    cache.prog_dyn_table = prog_dyn_table
    cache.prog_dyn_table_transitions = prog_dyn_table_transitions


def _sample_from_table_length(cache: AutomatonCache, in_the_language: bool = True) -> Trace:
    """
    Samples a trace uniformly at random from the automaton, using the
    statistics of the prog_dyn_table and prog_dyn_table_transitions.
    """
    automaton = cache.automaton
    length = cache.length
    prog_dyn_table = cache.prog_dyn_table
    prog_dyn_table_transitions = cache.prog_dyn_table_transitions

    assert length <= len(prog_dyn_table), "Invalid length: {}".format(length)
    assert length <= len(prog_dyn_table_transitions), "Invalid length: {}".format(length)

    final_states = []
    for state in range(automaton.num_states()):
        if in_the_language and automaton.state_is_accepting(automaton.state_from_number(state)):
            final_states.append(state)
        elif not in_the_language and not automaton.state_is_accepting(
            automaton.state_from_number(state)
        ):
            final_states.append(state)

    final_weights = [prog_dyn_table[length][state] for state in final_states]
    total_weights = sum(final_weights)
    assert total_weights > 0, "Invalid final weights: {}".format(final_weights)
    final_state = random.choices(final_states, weights=final_weights, k=1)[0]

    current_state = final_state
    trace = Trace(cache.ap, length)
    for l in range(length, 0, -1):
        transition_weights = [x[0] for x in prog_dyn_table_transitions[l][current_state]]
        assert sum(transition_weights) > 0, "Invalid transition weights: {}".format(
            transition_weights
        )
        transition = random.choices(
            prog_dyn_table_transitions[l][current_state], weights=transition_weights, k=1
        )[0][1]

        formula = transition.cond
        new_letter = sample_letter_from_BDD(formula, cache)
        assert len(new_letter) == len(cache.ap), "Invalid letter length: {} != {}".format(
            len(new_letter), len(cache.ap)
        )
        trace[l - 1] = new_letter

        current_state = transition.src

    assert current_state == automaton.get_init_state_number(), "Invalid first state: {}".format(
        current_state
    )
    assert len(trace) == length, "Invalid trace length: {} != {}".format(len(trace), length)

    return trace


def sample_distinct_traces(
    cache: AutomatonCache, number_traces: int = 1, in_the_language=True
) -> list[Trace]:
    """
    Samples `number_traces` distinct traces from the automaton
    obtained with function `prepare_cache`.
    `in_the_language` is a boolean indicating whether to sample
    traces in the language of the automaton or in the complement.
    """
    automaton = cache.automaton
    length = cache.length
    prog_dyn_table = cache.prog_dyn_table

    # Check that there are enough distinct traces in the automaton
    count_traces = 0
    for q in range(automaton.num_states()):
        if in_the_language and automaton.state_is_accepting(automaton.state_from_number(q)):
            count_traces += prog_dyn_table[length][q]
        elif not in_the_language and not automaton.state_is_accepting(
            automaton.state_from_number(q)
        ):
            count_traces += prog_dyn_table[length][q]

    if count_traces < number_traces:
        error_message = "Not enough distinct traces of length {} in the automaton (in_the_language={}): {} < {}".format(
            length, in_the_language, count_traces, number_traces
        )
        logging.error(error_message)
        assert False, error_message + "\n"

    # Sample the traces
    set_of_traces = set()
    while len(set_of_traces) < number_traces:
        trace = _sample_from_table_length(cache, in_the_language=in_the_language)
        set_of_traces.add(trace)

    return sorted(list(set_of_traces))


def prepare_cache(formula: str, ap: list[str], length: int) -> AutomatonCache:
    """
    Prepares the cache for the formula `formula` with the atomic
    propositions `ap` and the length of traces `length`.
    """
    automaton = compile(formula, ap=ap)
    logging.debug("Automaton\n%s", automaton.to_str("hoa"))
    assert spot.is_complete(automaton), "The automaton is not complete"
    assert spot.is_deterministic(automaton), "The automaton is not deterministic"
    assert sorted([str(a) for a in automaton.ap()]) == sorted(ap), (
        "The automaton does not have the same atomic propositions as the formula: {} != {}".format(
            sorted([str(a) for a in automaton.ap()]), sorted(ap)
        )
    )

    order = Ordering(automaton.get_dict(), ap)
    # Careful: automaton.ap() may follow an order different from ap (as not all
    # atomic propositions necessarily appear in the formula)
    # Throughout the code, we use the order of ap.
    assert len(ap) == order.len(), (
        "The length of the alphabet is not equal to the number of variables in the BDD order."
    )

    cache = AutomatonCache(formula, ap, automaton, length, order, {}, {}, {}, {})

    _prog_dyn_table_length(cache)

    return cache


if __name__ == "__main__":
    import ast
    import argparse
    import json
    import os

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Sample traces from an automaton.")
    parser.add_argument(
        "formula_file",
        type=str,
        help="Text file containing the 'formula; atomic propositions (as a list); name; repository, source'",
    )
    parser.add_argument(
        "number",
        type=int,
        help="Number of traces to sample (both for traces in and out the language).",
    )
    parser.add_argument("length", type=int, help="Length of the traces to sample.")
    parser.add_argument("-seed", type=int, default=42, help="Seed (default 42).")

    args = parser.parse_args()
    nb_traces = args.number
    length = args.length
    seed = args.seed
    logging.debug("Parsing arguments: %s", args)
    random.seed(seed)

    # Read the instance from the file
    file = args.formula_file
    logging.info("Processing file %s...", file)
    with open(file, "r") as f:
        file_content = f.read().split(";")
    if len(file_content) != 4:
        raise ValueError("Invalid file format. Expecting at least 4 ';'-separated fields.")

    # Declare the atomic propositions in the right order
    ap = ast.literal_eval(file_content[1])
    logging.debug("Atomic propositions: %s", str(ap))
    assert type(ap) == list, "The atomic propositions are not typeset as a Python list."
    assert all([type(a) == str for a in ap]), (
        "The atomic propositions are not typeset as Python strings."
    )

    formula = file_content[0]
    logging.debug("Formula: %s", formula)

    cache = prepare_cache(formula, ap, length)

    name = file_content[2]
    repository = file_content[3].split(",")[0]

    positive_traces = [
        trace.format_json(ap)
        for trace in sample_distinct_traces(cache, number_traces=nb_traces, in_the_language=True)
    ]
    negative_traces = [
        trace.format_json(ap)
        for trace in sample_distinct_traces(cache, number_traces=nb_traces, in_the_language=False)
    ]

    instance = {
        "positive_traces": positive_traces,
        "negative_traces": negative_traces,
        "smallest_known_formula": formula,
        "generating_formula": formula,
        "generating_seed": seed,
        "original_repository": repository,
        "name": file,
        "atomic_propositions": ap,
        "number_atomic_propositions": len(ap),
        "number_traces": len(positive_traces) + len(negative_traces),
        "number_positive_traces": len(positive_traces),
        "number_negative_traces": len(negative_traces),
        "max_length_traces": length,
        "trace_type": "finite",
    }

    # Save the traces to a file, replacing the file extension with .json
    file_name, _ = os.path.splitext(file)
    json_file_name = "{}_{}_{}_{}.json".format(file_name, nb_traces, length, seed)
    with open(json_file_name, "w") as f:
        json.dump(instance, f)
    logging.info("Saved traces to %s!\n", json_file_name)
