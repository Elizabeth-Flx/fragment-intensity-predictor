from model import Eval_Model
from data_formatting import df

import numpy as np

np.set_printoptions(edgeitems=60, linewidth=100000)

MAX_PRECURSOR = 3


model = Eval_Model(MAX_PRECURSOR)

tseq, tpre, tint = df.get_formatted_training  (2, MAX_PRECURSOR)
vseq, vpre, vint = df.get_formatted_validation(2, MAX_PRECURSOR)


print(tseq[1])
print(tpre[1])
print(tint[1])


'''print(df.generate_baseline(tint))
print(df.generate_baseline(vint))


print(len(tseq))
print(len(vseq))
'''

'''
print(df.spectral_angle(tint[0], tint[0]))

history = model.train_model([tseq,tpre], tint, 5, 128, ([vseq,vpre], vint))
print(history)



prediction = model.model.predict([vseq, vpre])

for i in range(400,450):
    print("")
    print("Datapoint: %s" % (i))
    print(vint[i])
    print(prediction[i])
    print(df.spectral_angle(vint[i], prediction[i]))


spec_ang = []
for i in range(len(vint)):
    spec_ang.append(df.spectral_angle(vint[i], prediction[i]))

print(sum(spec_ang)/len(spec_ang))

print(history)
'''


