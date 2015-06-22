#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pybrain.datasets import SupervisedDataSet
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, legend
import numpy
import time # Sera usada para testes de tempo
            # A aplicação final nao possui utilidade para isso
import Pyro4

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

controller = Pyro4.Proxy("PYRONAME:neural.network.controller")

controller.connectToSlaves()

# Carrega o arquivo com os dados. E recomendado usar arquivos '.data'
ds_array = numpy.loadtxt(sys.argv[1], delimiter=',')

# Assume que todos os campos sao input menos o ultimo
num_columns = ds_array.shape[1]
ds = SupervisedDataSet(num_columns - 1, 1)
ds.setField('input', ds_array[:,:-1])
ds.setField('target', ds_array[:,-1:])

controller.createNetwork(num_columns - 1,0,1,0,1,[7],1)

controller.createDataSet(ds)

controller.createTrainer()

start = time.time()
e = controller.trainNetwork(100) # Treina 100 epocas para teste
end = time.time()
dist_time = end - start

print "Tempo distribuído: {0}".format(dist_time) # Mostra o tempo de execucao do treino

plot(e, hold=True)
show()

del e[:]
start = time.time()
e = controller.sequentialTraining(100) # As mesmas 100 epocas em treinamento sequencial
end = time.time()
seq_time = end - start

print "Tempo Sequencial: {0}".format(seq_time)

plot(e, hold=True)
show()

print "Diferença de tempo: {0}".format(seq_time - dist_time)
