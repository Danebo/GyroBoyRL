#This script demonstrates the problems that arise from using python threads,
#the threads share a interpreter, and run sequentilly.
#Using processes solves this problem. Also shows multiprocess communication.
#Written by Rasmus Loft and Andreas Danebo Jensen

import time
from threading import Thread
from multiprocessing import Process

def getTime():
    return int(round(time.time()*1000))

def count(n):
    print'counting'
    while n > 0:
		n -=1

#sequantial test		
print('sequential test, count to 5000000 x4')
Start_time = getTime()
count(5000000)
count(5000000)
count(5000000)
count(5000000)
print(getTime()-Start_time)
#count(10000000)

#multi thread test
print('multi thread test, 4 threads counting')
Start_time = getTime()
t1 = Thread(target=count,args=(5000000,))
t1.start()
t2 = Thread(target=count,args=(5000000,))
t2.start()
t3 = Thread(target=count,args=(5000000,))
t3.start()
t4 = Thread(target=count,args=(5000000,))
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()
print(getTime()-Start_time)

#multi process
print('multi process test, 4 process counting')
Start_time = getTime()
p1 = Process(target=count,args=(5000000,))
p1.start()
p2 = Process(target=count,args=(5000000,))
p2.start()
p3 = Process(target=count,args=(5000000,))
p3.start()
p4 = Process(target=count,args=(5000000,))
p4.start()
p1.join() 
p2.join()
p3.join() 
p4.join()
print(getTime()-Start_time)

#multi process producer/consumer
print('multi process producer/consumer test, 4 process counting')
import multiprocessing
import time

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):
    def __call__(self):
        count(5000000)
        return 'done'

Start_time = getTime()	
#Establish communication queues
tasks = multiprocessing.JoinableQueue()
results = multiprocessing.Queue()

#Start consumers
num_consumers = 4
consumers = [ Consumer(tasks, results)
              for i in xrange(num_consumers) ]
for w in consumers:
    w.start()

#Enqueue jobs
num_jobs = 4
for i in xrange(num_jobs):
    tasks.put(Task())

#Add a poison pill for each consumer
for i in xrange(num_consumers):
    tasks.put(None)

#Wait for all of the tasks to finish
tasks.join()

#Start printing results
while num_jobs:
    result = results.get()
    print 'Result:', result
    num_jobs -= 1
print(getTime()-Start_time)