# adapted from https://github.com/whitemech/finite-synthesis-datasets

import argparse

def Not(form):
    return "!" + form

def BigAnd(forms):
    if not forms:
        return "true"
    else:
        return "(" + " && ".join(forms) + ")"

def And(form1, form2):
    return BigAnd([form1, form2])

def And3(form1, form2, form3):
    return BigAnd([form1, form2, form3])

def BigOr(forms):
    if not forms:
        return "false"
    else:
        return "(" + " || ".join(forms) + ")"

def IfThen(form1, form2):
    return "(" + form1 + " -> " + form2 + ")"

def Iff(form1, form2):
    return "(" + form1 + " <-> " + form2 + ")"

def Next(form):
    return "X[!] " + form

def Globally(form):
    return "G " + form

def Until(form1, form2):
    return "(" + form1 + " U " + form2 + ")"

def select(player, i):
    return "select_" + player + "_" + str(i)

def change(player, s):
    return "change_" + player + "_" + str(s)

def turn(player):
    return "turn_" + player

def heap(i, s):
    return "heap_" + str(i) + "_" + str(s)


def rules(player, n, m):
    return [IfThen(turn(player), BigOr([select(player, i) for i in range(n)]))] + \
           [IfThen(select(player, i), Not(select(player, j)))
            for i in range(n)
            for j in range(i)] + \
           [IfThen(change(player, s), Not(change(player, t)))
            for s in range(m)
            for t in range(s)] + \
           [IfThen(And3(BigOr([Not(heap(j, 0)) for j in range(n)]),
                        heap(i, s),
                        Next(select(player, i))),
                   BigOr([Next(change(player, s_prime)) for s_prime in range(s)]))
            for i in range(n)
            for s in range(m + 1)]

def main(n, m):
    env = "env"
    sys = "sys"

    inputs = [select(env, i) for i in range(n)] + \
             [change(env, s) for s in range(m)]

    outputs = [select(sys, i) for i in range(n)] + \
              [change(sys, s) for s in range(m)] + \
              [turn(sys), turn(env)] + \
              [heap(i, s) for i in range(n) for s in range(m + 1)]

    preset = [turn(sys), Not(turn(env))] + \
             [IfThen(Not(select(sys, i)), heap(i, m)) for i in range(n)] + \
             [IfThen(select(sys, i), BigOr([change(sys, s) for s in range(m)]))
              for i in range(n)]

    require = rules(env, n, m)

    asserts = rules(sys, n, m) + \
              [Iff(turn(sys), Not(turn(env)))] + \
              [IfThen(Next(turn(sys)), turn(env))] + \
              [IfThen(Next(turn(env)), turn(sys))] + \
              [IfThen(heap(i, s), Not(heap(i, t)))
               for i in range(n)
               for s in range(m + 1)
               for t in range(s)] + \
              [IfThen(And3(turn(p), select(p, i), change(p, s)), heap(i, s))
               for p in [env, sys]
               for i in range(n)
               for s in range(m)] + \
              [IfThen(And3(Next(turn(p)), Next(Not(select(p, i))), heap(i, s)),
                      Next(heap(i, s)))
               for p in [env, sys]
               for i in range(n)
               for s in range(m + 1)]

    guarantee = Until(BigOr([Not(heap(i, 0)) for i in range(n)]),
                      And(turn(env), BigAnd([heap(i, 0) for i in range(n)])))

    env_assumption = Globally(BigAnd(require))

    monolithic = And(BigAnd(preset),
                     IfThen(env_assumption,
                            And(Globally(BigAnd(asserts)), guarantee)))

    with open("Nim/heaps=" + str(n) + "tokens=" + str(m) + ".txt", "w") as f:
        f.write(monolithic + ";" + str(inputs + outputs) + ";" + "nim_heaps=" + str(n) + "tokens=" + str(m) + ";" + "https://github.com/whitemech/finite-synthesis-datasets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an LTL learning formula for Nim benchmark family.")
    parser.add_argument("-heaps", type=int, default=2, help="Number of heaps.")
    parser.add_argument("-tokens", type=int, default=1, help="Number of tokens.")

    args = parser.parse_args()

    main(args.heaps, args.tokens)
