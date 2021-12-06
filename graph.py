import matplotlib.pyplot as plt

from multiprocessing.connection import Listener
import json

def plot(t, data):
	plt.clf()
	plt.legend()
	for key in data:
		plt.plot(t, data[key],'',label=key)

	plt.legend(loc='best')
	plt.pause(0.01)

address = ('localhost', 6000)
listener = Listener(address, authkey=bytes('graph', 'utf-8'))
conn = listener.accept()
print("accepted")
while True:
    msg = conn.recv()
    decoded_json = json.loads(msg)
    plot(decoded_json[0], decoded_json[1])

listener.close()