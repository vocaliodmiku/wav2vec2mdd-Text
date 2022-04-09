LARGE_set = eval(open('LARGE_set', 'r').readline())
LV60_set =  eval(open('LV60_set', 'r').readline())
XLSR_set =  eval(open('XLSR_set', 'r').readline())
XLSR33_set =  eval(open('XLSR33_set', 'r').readline())
XLSR66_set =  eval(open('XLSR66_set', 'r').readline())
print("LARGE_set:", len(LARGE_set))
print("LV60_set:", len(LV60_set))
print("XLSR_set:", len(XLSR_set))
print("XLSR33_set:", len(XLSR33_set))
print("XLSR66_set:", len(XLSR66_set))

def show(a,b):
    res = {}
    for i,j in a.items():
        if i in b.keys():
            if j == b[i]:
                continue
            else:
                res[i] = j-b[i]
    return res

c,d = show(XLSR_set, LARGE_set), show(XLSR_set, LV60_set)
print(c)
print(d)


