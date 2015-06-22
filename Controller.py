#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, TanhLayer, GaussianLayer, SoftmaxLayer, BiasUnit, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, legend
from copy import deepcopy
from multiprocessing import Process
from random import shuffle
import numpy
import Pyro4
import subprocess

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
daemon = Pyro4.Daemon()#host='192.168.0.1') # Mudar para o host atual
ns = None

class Controller(object):
    def __init__(self):
        self.net = FeedForwardNetwork()
        self.threadList = []


    def connectToSlaves(self):
        global daemon
        global ns
        del self.threadList[:]
        for slave, slave_uri in ns.list(prefix="neural.node.").items():
            self.threadList.append(Pyro4.Proxy(slave_uri))


    # Cria a rede neural a ser usada. Essa função sera duplicada nos codigos dos escravos
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

        jobs = []
        for t in self.threadList:
            p = Process(target=t.createNetwork, args=(inLayer, inLType, outLayer, outLType, hLayerNum, hiddenLayers, hLayersType, bias, outPutBias))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        del jobs[:]
        for t in self.threadList:
            p = Process(target=t.setParameters, args=(self.net.params,))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()


    def createDataSet(self, ds):
        inp = ds.indim
        targ = ds.outdim

        self.ds = SupervisedDataSet(inp, targ)

        dsValues = []

        # Guarda os valores do dataset
        for temp in ds._provideSequences():
            dsValues.append(temp)

        shuffle(dsValues) # Embaralha os valores do dataset

        for i in dsValues:
            self.ds.addSample(i[0][0],i[0][1])
        
        numProp = len(self.threadList)
        dsProportion = 1.0/numProp
        
        # Cria os datasets nos Slaves
        if numProp > 1:
            tempDs = ds
            for t in self.threadList[:-1]:
                tempDs = tempDs.splitWithProportion(dsProportion)
                t.createDataSet(tempDs[0])
                numProp -=1
                dsProportion = 1.0/numProp
                tempDs = tempDs[1]
            self.threadList[-1].createDataSet(tempDs)
        else:
            self.threadList[0].createDataSet(ds)


    def createTrainer(self, learnrate=0.01, ldecay=1.0, momentum=0.0, batchlearn=False, wdecay=0.0):
        jobs=[]
        for t in self.threadList:
            p = Process(target=t.createTrainer, kwargs={'learnrate': learnrate, 'ldecay': ldecay, 'momentum': momentum, 'batchlearn': batchlearn, 'wdecay': wdecay})
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        self.trainer = BackpropTrainer(self.net, self.ds, learningrate=learnrate, lrdecay=ldecay, momentum=momentum, batchlearning=batchlearn, weightdecay=wdecay)


    def trainNetwork(self, numEpochs=1):

        # Declarações das listas que serão usadas no loop
        # Declaradas na ordem em que aparecerão pela primeira vez no código
        jobs = []
        temp = []
        #dsValues = []
        e = []

        for i in range(0,numEpochs):
            del jobs[:]

            # Realiza o treinamento em cada Slave
            for t in self.threadList:
                p = Process(target=t.trainNetwork)
                jobs.append(p)
                p.start()
            
            for p in jobs:
                p.join()
            
            ############### Atualiza os Parâmetros ###############
            # Variável que realizará a atualização dos parâmetros
            updatedParameters = [0.0 for j in self.net.params]

            # Os parâmetros de cada Slave são somados
            for t in self.threadList:
                del temp[:]
                temp = t.getParameters()
                updatedParameters = [x+y for x,y in zip(updatedParameters, temp)]

            # A média é calculada e os Slaves são atualizados
            updatedParameters = [j/len(self.threadList) for j in updatedParameters]
            updatedParameters = numpy.array(updatedParameters)
            self.net._setParameters(updatedParameters)

            del jobs[:]
            for t in self.threadList:
                p = Process(target=t.setParameters, args=(updatedParameters,))
                jobs.append(p)
                p.start()

            for p in jobs:
                p.join()

            # Determina o erro para a época atual
            error = 0.0
            for i,t in self.ds:
                value = self.net.activate(i)
                localError = t - value
                error += localError * localError
            error = error/len(self.ds)
            e.append(error)

        return e


    def sequentialTraining(self, numEpochs=1):
        e = []
        for i in range(0,numEpochs):
            self.trainer.train()

            #Determina o erro para a época atual
            error = 0.0
            for i,t in self.ds:
                value = self.net.activate(i)
                localError = t - value
                error += localError * localError
            error = error/len(self.ds)
            e.append(error)

        return e

    def activateNetwork(self, data):
        value = self.net.activate(data)
        return value

    def saveNetwork(self, fileName):
        NetworkWriter.writeToFile(self.net, fileName + '.xml')

    def loadNetwork(self, fileName):
        del self.net
        jobs = []

        self.net = NetworkReader.readFrom(fileName + '.xml')

        for t in self.threadList:
            p = Process(target=t.loadNetwork, args=(self.net,))
            jobs.append(p)
            p.start()

def main():
    global daemon
    global ns
    controller = Controller()
    uri = daemon.register(controller)
    ns = Pyro4.locateNS()#host='192.168.0.1') # Mudar para o host atual
    ns.register("neural.network.controller", uri)
    daemon.requestLoop()

if __name__ == '__main__':
    main()
