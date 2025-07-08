import argparse
import random
import ast

def main(L:int, m:int, seed:int) -> (dict,list):
    random.seed(seed)

    atomic_propositions = [f"a{i}" for i in range(m)]
    with open("../Fixed_Formulas/formulas.txt", "r") as f:
        out = f.readlines()
    basic_formulas = [(x.split(";")[0],ast.literal_eval(x.split(";")[1])) for x in out]

    conjuncts = random.choices(basic_formulas,k = L)
    randomised_conjuncts = []
    for (formula,variables) in conjuncts:
        new_variables = random.sample(atomic_propositions, k = len(variables))
        for i, v in enumerate(variables):
            formula = formula.replace(v,new_variables[i])
        randomised_conjuncts.append(formula[1:-1])

    formula = " && ".join(randomised_conjuncts)
    param = "conjuncts=" + str(L) + "_atomic_propositions=" + str(m) + "_seed=" + str(seed)
    with open("Random_Conjuncts_from_Basis/" + param + ".txt", "w") as f:
        name = "random_conjuncts_from_basis_" + param
        f.write(formula + ";" + str(atomic_propositions) + ";" + name + ";" + "https://arxiv.org/abs/1705.08426")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an LTL formula for RandomConjunctionsFromBasis benchmark family.")
    parser.add_argument("-L", type=int, default=4, help="Number of conjuncts.")
    parser.add_argument("-m", type=int, default=5, help="Number of atomic propositions.")
    parser.add_argument("-seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    main(args.L, args.m, args.seed)
