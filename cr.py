import numpy as np
from CSVParse import CSVParse

csv = CSVParse()
csv.read('currency.csv')

data = np.array(csv.data[1:])

diff = []
for i in xrange(1,len(data)):
    diff.append(data[i] - data[i-1])

#target currency: USD, JPY, CNY, EUR, GBP, CHF, CAD, AUD
target = 'EUR'
score = []
for i in xrange(1,len(diff)):
    x = 1 if (diff[i][csv.data[0].index(target)]>0) else  -1
    score.append(x)

from sklearn.neural_network import MLPClassifier

for i in xrange(10,1000,10):
    mlp = MLPClassifier(solver='sgd', learning_rate_init=0.1, alpha=0, batch_size=1900,
                        activation='logistic', random_state=10, max_iter=2000,
                        hidden_layer_sizes=i, momentum=0)
    avg = 0
    for j in xrange(1,30):
        mlp.fit(diff[0:1900],score[0:1900])
        avg += mlp.score(diff[1901:2050],score[1901:2050])
    print i, avg/30.0

