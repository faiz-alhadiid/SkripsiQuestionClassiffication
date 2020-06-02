
file = open('debug.txt', 'w')

def writeln(*obj):
    obj = " ".join([str(x) for x in obj])
    # file.writelines(obj+"\n")

def flush():
    pass
    # file.flush()