"""
Code by Rasmus Loft and Andreas Danebo Jensen
"""

import multiprocessing
import time
import state

class PIDState(multiprocessing.Process):

    debug = False
   
    #make a queue for recieving sensor data, will only use newest entry
    def __init__(self, task_queue, stateQueue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.stateQueue = stateQueue
        
    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
			#get newest reading, check for poison
            while (self.task_queue.empty()==False and (next_task is not None)):
                next_task =self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                if (self.debug):
                    print '%s: Exiting' % proc_name
                break
            self.state = next_task()
            if (self.debug):
                print("show state")
                self.state.showMe()
            self.stateQueue.put(self.state)
        return