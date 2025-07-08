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

def WeakNext(form):
    return "X " + form

def Globally(form):
    return "G " + form

def Eventually(form):
    return "F " + form

def Until(form1, form2):
    return "(" + form1 + " U " + form2 + ")"

def init_counter(i):
    return "init_counter_" + str(i)

def counter(i):
    return "counter_" + str(i)

def carry(i):
    return "carry_" + str(i)

def main(n):
    inc = "inc"

    inputs = [init_counter(i) for i in range(n)] + \
             [inc]

    outputs = [counter(i) for i in range(n)] + \
              [carry(i) for i in range(n)]

    preset = [And(IfThen(Next(counter(i)), init_counter(i)),
                  IfThen(init_counter(i), WeakNext(counter(i)))) for i in range(n)]

    require = [IfThen(Not(inc), Next(inc))]

    asserts = [IfThen(Next(carry(0)), inc)] + \
              [IfThen(inc, WeakNext(carry(0)))] + \
              [IfThen(Next(carry(i)), And(counter(i - 1),
                                          Next(carry(i - 1))))
               for i in range(1, n)] + \
              [IfThen(And(counter(i - 1),
                          WeakNext(carry(i - 1))),
                      WeakNext(carry(i)))
               for i in range(1, n)] + \
              [And(IfThen(Next(counter(i)),
                          Not(Iff(counter(i), Next(carry(i))))),
                   IfThen(Not(Iff(counter(i), WeakNext(carry(i)))),
                          WeakNext(counter(i))))
               for i in range(n)]

    guarantee = Next(Eventually(BigAnd([Not(counter(i)) for i in range(n)])))

    env_assumption = Globally(BigAnd(require))

    monolithic = And(BigAnd(preset),
                     IfThen(env_assumption,
                            And(Next(Globally(BigAnd(asserts))), guarantee)))

    with open("SingleCounter/" + str(n) + ".txt", "w") as f:
        f.write(monolithic + ";" + str(inputs + outputs) + ";" + "singlecounter_bits=" + str(n) + ";" + "https://github.com/whitemech/finite-synthesis-datasets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an LTL formula for SingleCounter benchmark family.")
    parser.add_argument("-N", type=int, default=4, help="Number of bits.")

    args = parser.parse_args()

    main(args.N)