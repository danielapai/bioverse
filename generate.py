# System modules
import multiprocessing as mp
import traceback

# Bioverse modules
from classes import Stopwatch
import utils

def generator(program,s=None,timed=False):
    if s is None: s = {}
    if timed:
        timer = Stopwatch()
        timer.start()
    for i in range(len(program)):
        func = utils.import_function_from_file(program[i][0],program[i][1])
        arguments = program[i][-1]
        args = (arguments[key][0] for key in arguments.keys())
        try:
            func(s,*args)
            if timed: timer.mark(program[i][0])
        except:
            traceback.print_exc()
            print("\n!!! The program failed at step {:d}: {:s} !!!".format(i,program[i][0]))
            print("!!! Returning incomplete simulation !!!")
            return s
    if timed:
        print("Timing results:")
        timer.read()
    return s

def generator_multi(program,N_proc=4,idx_split=1):
    # Run the program as usual up until the split point specified by idx_split
    s = generator(p,program[:idx_split])

    # Now split the universe into (approx) equal sizes chunks
    ss = s.split(N_proc)

    # Update the parameters dict to have the appropriate number of planets
    ps = []
    for i in range(N_proc):
        ps.append(p.copy())
        ps[i]['N'] = len(ss[i])

    # Continue to run each chunk in parallel through the program
    q = mp.Queue()
    def run(i):
        ret = generator(ps[i],program[idx_split:],ss[i])
        q.put(ret)
    procs = [mp.Process(target=run,args=(i,)) for i in range(N_proc)]
    for proc in procs:
        proc.start()
    results = [q.get() for proc in procs]
    for proc in procs:
        proc.join()
    
    # Combine these chunks into a single result
    print("Combining results from {:d} processes...".format(N_proc))
    s = {}
    for result in results:
        s.append(result)

    return s

























