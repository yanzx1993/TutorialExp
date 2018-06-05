import multiprocessing as mp
import random as rd

NUM_AGENT = 4

def master(param_queue,exp_queue):
    a=rd.random()
    b=rd.random()
    for i in range(NUM_AGENT):
        param_queue[i].put([a,b])
    print(a,b)
    for i in range(NUM_AGENT):
        sum=exp_queue[i].get()
        print(sum)



def worker(param_queue,exp_queue):
    a,b=param_queue.get()
    factor=rd.random()
    suma=(a+b)*factor
    exp_queue.put(suma)



def main():
    param_queue = []
    exp_queue = []
    for i in range(NUM_AGENT):
        param_queue.append(mp.Queue(1))
        exp_queue.append(mp.Queue(1))
    coord=mp.Process(target=master,args=(param_queue,exp_queue))
    coord.start()
    agent=[]
    for i in range(NUM_AGENT):
        agent.append(mp.Process(target=worker,args=(param_queue[i],exp_queue[i])))
    for i in range(NUM_AGENT):
        agent[i].start()
    coord.join()

main()
