import threading
import numpy as np

class myThread (threading.Thread):
	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter

	def run(self):
		print("Starting " + self.name)
		# Get lock to synchronize threads
		threadLock.acquire()
		long_operation()
		# Free lock to release next thread
		threadLock.release()

def long_operation(n = 1000, r = 1000):
	M = np.random.rand(n, n)
	for i in range(r):
		M = np.matmul(M, np.random.rand(n, n))

long_operation()

# threadLock = threading.Lock()
# threads = []

# # Create new threads
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)

# # Start new Threads
# thread1.start()
# thread2.start()

# # Add threads to thread list
# threads.append(thread1)
# threads.append(thread2)

# # Wait for all threads to complete
# for t in threads:
	# t.join()
# print("Exiting Main Thread")
