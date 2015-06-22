#!/usr/bin/env python
# -*- coding: utf-8 -*-

from array import array
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer, GaussianLayer, BiasUnit, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, legend
import Pyro4

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config. SERIALIZERS_ACCEPTED.add('pickle')

class Slave(object):
    def __init__(self):
        self.net = FeedForwardNetwork()

    def createNetwork(self, inLayer, inLType, outLayer, outLType, hLayerNum, hiddenLayers, hLayersType, bias=True, outPutBias=True):

        del self.net
        self.net = FeedForwardNetwork()

        if bias:
            # Definição da camada de entrada
            if inLType == 0:
                self.net.addInputModule(LinearLayer(inLayer,name='in'))
            elif inLType == 1:
                self.net.addInputModule(SigmoidLayer(inLayer,name='in'))
            elif inLType == 2:
                self.net.addInputModule(TanhLayer(inLayer,name='in'))
            elif inLType == 3:
                self.net.addInputModule(SoftmaxLayer(inLayer,name='in'))
            elif inLType == 4:
                self.net.addInputModule(GaussianLayer(inLayer,name='in'))

            # Definição das camadas escondidas
            self.hiddenLayers = []
            if hLayersType == 0:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(LinearLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 1:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(SigmoidLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 2:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(TanhLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 3:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(SoftmaxLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 4:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(GaussianLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])

            # Definição da camada de saída
            if outLType == 0:
                self.net.addOutputModule(LinearLayer(outLayer,name='out'))
            elif outLType == 1:
                self.net.addOutputModule(SigmoidLayer(outLayer,name='out'))
            elif outLType == 2:
                self.net.addOutputModule(TanhLayer(outLayer,name='out'))
            elif outLType == 3:
                self.net.addOutputModule(SoftmaxLayer(inLayer,name='out'))
            elif outLType == 4:
                self.net.addOutputModule(GaussianLayer(outLayer,name='out'))

            # Criação do Bias
            self.net.addModule(BiasUnit(name='networkBias'))

            # Conexão entre as diversas camadas
            if self.hiddenLayers:
                self.net.addConnection(FullConnection(self.net['in'], self.hiddenLayers[0]))
                for h1, h2 in zip(self.hiddenLayers[:-1], self.hiddenLayers[1:]):
                    self.net.addConnection(FullConnection(self.net['networkBias'],h1))
                    self.net.addConnection(FullConnection(h1,h2))
                if outPutBias:
                    self.net.addConnection(FullConnection(self.net['networkBias'],self.net['out']))
                self.net.addConnection(FullConnection(self.hiddenLayers[-1],self.net['out']))
            else:
                if outPutBias:
                    self.net.addConnection(FullConnection(self.net['networkBias'],self.net['out']))
                self.net.addConnection(FullConnection(self.net['in'],self.net['out']))
        else:
            # Definição da camada de entrada
            if inLType == 0:
                self.net.addInputModule(LinearLayer(inLayer,name='in'))
            elif inLType == 1:
                self.net.addInputModule(SigmoidLayer(inLayer,name='in'))
            elif inLType == 2:
                self.net.addInputModule(TanhLayer(inLayer,name='in'))
            elif inLType == 3:
                self.net.addInputModule(SoftmaxLayer(inLayer,name='in'))
            elif inLType == 4:
                self.net.addInputModule(GaussianLayer(inLayer,name='in'))

            # Definição das camadas escondidas
            self.hiddenLayers = []
            if hLayersType == 0:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(LinearLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 1:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(SigmoidLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 2:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(TanhLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 3:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(SoftmaxLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])
            elif hLayersType == 4:
                for i in range(0, hLayerNum):
                    self.hiddenLayers.append(GaussianLayer(hiddenLayers[i]))
                    self.net.addModule(self.hiddenLayers[i])

            # Definição da camada de saída
            if outLType == 0:
                self.net.addOutputModule(LinearLayer(outLayer,name='out'))
            elif outLType == 1:
                self.net.addOutputModule(SigmoidLayer(outLayer,name='out'))
            elif outLType == 2:
                self.net.addOutputModule(TanhLayer(outLayer,name='out'))
            elif outLType == 3:
                self.net.addOutputModule(SoftmaxLayer(inLayer,name='out'))
            elif outLType == 4:
                self.net.addOutputModule(GaussianLayer(outLayer,name='out'))

            if self.hiddenLayers:
                self.net.addConnection(FullConnection(self.net['in'], self.hiddenLayers[:1]))
                for h1, h2 in zip(self.hiddenLayers[:-1], self.hiddenLayers[1:]):
                    self.net.addConnection(FullConnection(h1,h2))
                self.net.addConnection(FullConnection(self.hiddenLayers[-1:],self.net['out']))
            else:
                self.net.addConnection(FullConnection(self.net['in'],self.net['out']))

        # Termina de construir a rede e a monta corretamente
        self.net.sortModules()

    def setParameters(self, parameters):
        self.net._setParameters(parameters)

    def getParameters(self):
        return self.net.params.tolist()

    def createDataSet(self, ds):
        inp = ds.indim
        targ = ds.outdim

        self.ds = SupervisedDataSet(inp, targ)

        for i,t in ds:
            self.ds.addSample(i,t)

    def updateDataSet(self, ds):
        self.ds.clear(True)
        for i,t in ds:
            self.ds.addSample(i,t)
        self.trainer.setData(self.ds)

    def createTrainer(self, learnrate=0.01, ldecay=1.0, momentum=0.0, batchlearn=False, wdecay=0.0):
        self.trainer = BackpropTrainer(self.net, self.ds, learningrate=learnrate, lrdecay=ldecay, momentum=momentum, batchlearning=batchlearn, weightdecay=wdecay)

    def trainNetwork(self):
        self.trainer.train()

    def loadNetwork(self, net):
        del self.net
        self.net = net

def main():
    slave = Slave()
    daemon = Pyro4.Daemon()#host='192.168.0.2') # Mudar para o host atual. Cada slave deve ter um host único.
    slave_uri = daemon.register(slave)
    ns = Pyro4.locateNS()#host='192.168.0.1') # Mudar para o host do servidor de nomes. Todos os Slaves devem se conectar ao mesmo host
    numThreads = len(ns.list())-1
    ns.register("neural.node.slave{0}".format(numThreads), slave_uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()
