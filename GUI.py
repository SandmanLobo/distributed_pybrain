#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys
from easygui import *
from pybrain.datasets import SupervisedDataSet
import pylab
import numpy
import Pyro4

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

while 1:
    if os.path.isfile('NNIP.txt'):
        with open('NNIP.txt', 'r') as ipInfo:
            IP = ipInfo.read()
    else:
        msg = 'Insira o IP onde a rede neura deve ser encontrada'
        title = 'Conexão'
        IP = enterbox(msg, title)
        ipInfo = open('NNIP.txt', 'w+')
        ipInfo.write(IP)

    ipInfo.close()

    controller = Pyro4.Proxy("PYRONAME:neural.network.controller")

    msg = 'Escolha de função'
    title = 'Escolha de função'
    choices = {1:'Criar e treinar rede neural', 2:'Ativar rede neural salva'}
    choice = choicebox(msg, title, choices.values())

    if choice == choices[1]:
        title = 'Escolha do DataSet'
        filetypes = ['*.data', ['*.csv', '*.txt', 'Data files']]
        filepath = fileopenbox(title=title, filetypes=filetypes)
        
        ds_array = numpy.loadtxt(filepath, delimiter=',')

        #Assume que todos os campos são input menos o último
        num_columns = ds_array.shape[1]
        ds = SupervisedDataSet(num_columns-1, 1)
        ds.setField('input', ds_array[:,:-1])
        ds.setField('target', ds_array[:,-1:])
        
        del choices

        msg = 'Defina o tipo de função de ativação da camada de entrada'
        title = 'Parâmetros'
        choices = {'Linear':0,'Sigmoid':1,'Tangente Hiperbólica':2,'Softmax':3,'Gaussiana':4}
        choice = choicebox(msg, title, choices.keys())
        inLType = choices[choice]

        msg = 'Defina o número de camadas intermediárias'
        hLayerNum = integerbox(msg, title)
        
        hiddenLayers = []
        for i in range(hLayerNum):
            msg = 'Defina o número de neurônios na {0}ª camada intermediária'.format(i+1)
            temp = integerbox(msg, title)
            hiddenLayers.append(temp)

        msg = 'Defina o tipo da função de ativação das camadas intermediárias'
        choice = choicebox(msg, title, choices.keys())
        hLayersType = choices[choice]

        msg = "Defina o tipo da função de ativação da camada de saída"
        choice = choicebox(msg, title, choices.keys())
        outLType = choices[choice]

        msg = 'Usar bias?'
        bias = boolbox(msg, title, ["Sim", "Não"])

        msg = 'Usar bias na camada de saída?'
        outPutBias = boolbox(msg, title, ["Sim", "Não"])

        controller.connectToSlaves()

        controller.createNetwork(num_columns-1, inLType, 1, outLType, hLayerNum, hiddenLayers, hLayersType, bias, outPutBias)

        controller.createDataSet(ds)

        msg = 'Defina a taxa de aprendizado'
        title = 'Função de aprendizado'
        temp = enterbox(msg, title, '0.01')
        learnrate = float(temp)
        
        controller.createTrainer(learnrate=learnrate)

        msg = 'Defina o número de épocas a ser treinado'
        epoch = integerbox(msg, title, '1', upperbound=60000)

        print 'Por favor, aguarde. Rede em treinamento...'
        
        e = controller.trainNetwork(epoch)
        
        pylab.xlabel(u'Épocas')
        pylab.ylabel(u'Erro médio quadrático')
        pylab.plot(e, hold=True)
        pylab.show()

        msg = 'Salvar rede treinada?'
        title = 'Salvando'

        if boolbox(msg, title, ["Sim", "Não"]):
            msg = 'Digite o nome da rede'
            name = enterbox(msg,title)

            controller.saveNetwork(name)

        sys.exit(0)
    elif choice == choices[2]:
        msg = 'Insira o nome da rede a ser carregada'
        title = 'Carregar Rede'
        name = enterbox(msg,title)

        controller.loadNetwork(name)

        msg = 'Insira os valores a serem usado na entrada da rede, separados por vírgula'
        title = 'Definição da Entrada'
        rawValues = enterbox(msg, title)
        separatedValues = rawValues.split(',')
        inputValues = []
        for value in separatedValues:
            inputValues.append(float(value))

        outputValue = controller.activateNetwork(inputValues)

        msg = 'Valor de saída da rede: {0}'.format(outputValue)
        title = 'Saída'
        msgbox(msg, title, ok_button='OK')

        sys.exit(0)
    else:
        sys.exit(0)
