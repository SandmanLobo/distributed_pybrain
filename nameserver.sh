#!/bin/bash
export PYRO_SERIALIZERS_ACCEPTED=serpent,json,marshal,pickle
pyro4-ns #-n 192.168.0.1 # Mudar o IP para o da máquina atual
