import matplotlib.pyplot as plt
import numpy as np
import os

RNN = 'RNN'
GRU= 'GRU'
TR = 'TR'


dirs = [RNN,GRU,TR ]
labels = ['RNN','GRU','Transformer']
colors = ['C'+str(i) for i in range(len(dirs))]
i=0
times=np.arange(35)+1
for d in dirs:
    lc_path = os.path.join(d, 'model_losses.npy')
    model_loss = np.load(lc_path)[()]
    plt.plot(times, model_loss, '--^', color=colors[i], alpha=0.9, label=labels[i])
    i+=1
plt.legend()
plt.title("Average loss per validation sequences")
plt.ylabel("Validation loss")
plt.xlabel("Time step")

plt.savefig('5.l.png')
plt.clf()
plt.close()
