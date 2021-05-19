def subset(set_a, set_b):
    ''' Returns True if the fluents in set_a are also in set_b'''
    in_set_b = True
    for fluent_a in set_a:
        in_set_b = in_set_b and fluent_a.in_state(set_b)
    return in_set_b
        