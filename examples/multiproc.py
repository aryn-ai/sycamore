import time

from multiprocessing import Pool

def count(max):
    i = 0
    while i < max:
        i = i + 1
    print("count", i)
    return {"eric": "abc", "bin": b"0000", "count": i}

def wtcount(max):
    wall_time(lambda: count(max))

def mpcount(max):
    with Pool(16) as p:
#        print(p.map(count, [max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max]))
#        print(p.map(wtcount, [max, max, max, max, max, max, max, max]))
        print(p.map(wtcount, [max, max, max, max]))


def make_big_list(count, elem_size):
    ret = []
    for i in range(count):
        base_str = str(i)
        ret.append(base_str * int(elem_size / len(base_str)))
    return ret

def count_list_elems(l):
    total = 0
    for i in l:
        for j in range(len(i)):
            total = total + int(i[j])
    return total

def rep(num, fn):
    for i in range(num):
        fn()
        
def count_list(l):
    total = 0
    for i in l:
        total = total + len(i)

def wall_time(fn):
    # wall_time(lambda: time.sleep(1))
    start = time.time_ns()
    ret = fn()
    end = time.time_ns()
    print("Elapsed time:", (end - start) / 1.0e9)
    return ret

big_list = wall_time(lambda: make_big_list(1000*1000, 100))
nreps = 100

def rep_count_list(foo):
    wall_time(lambda: rep(nreps, lambda: count_list(big_list)))

def mprepcountlist(psize):
    l = []
    for i in range(psize):
        l.append(i)
    with Pool(psize) as p:
        print(p.map(rep_count_list, l))

#wall_time(lambda: count(500* 1000* 1000))
#wall_time(lambda: mpcount(500 * 1000 * 1000))

#wall_time(lambda: print(count_list_elems(big_list)))
wall_time(lambda: rep(nreps, lambda: count_list(big_list)))
wall_time(lambda: mprepcountlist(12))
