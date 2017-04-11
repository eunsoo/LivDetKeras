# -*- coding: utf-8 -*-

from keras.models import Model, load_model
import json
import matplotlib.pyplot as plt
#%%
logs = {'acc': [0.93907407407407406, 0.97805555555555557, 0.98370370370370375],
 'loss': [0.19464742978413899, 0.075911519196298391, 0.056960223786256932],
 'val_acc': [0.99250000000000005, 0.99750000000000005, 0.99250000000000005],
 'val_loss': [0.08624507377545039, 0.026560386419296266, 0.021675664447247982]}
#%%
# summarize history for accuracy
fig = plt.Figure()
fig.set_canvas(plt.gcf().canvas)
plt.subplot(211)
plt.plot(logs['acc'])
plt.plot(logs['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(212)
plt.plot(logs['loss'])
plt.plot(logs['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./fuckingSituation.png')
#%%


#plt.show()
with open('filefucking.json', 'w') as f:
    f.write(json.dumps(logs))

fuckyou =[logs, logs]
fucking = json.dumps(fuckyou)

    
fucking = json.dumps(logs)
fuck = json.loads(fucking)
