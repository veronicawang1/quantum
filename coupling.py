def coupleState(state1, state2):
    # 1 2
    # 1 3
    # 2 3
    coupled = (compareState(state1[0], state2[1]) + 
               compareState(state1[0], state2[2]) + 
               compareState(state1[1], state2[2]))
    return coupled

def compareState(state1, state2):
    if state1 == state2:
        return 0
    else:
        return 1