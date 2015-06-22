#!/bin/bash
wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python
sudo apt-get install python-pip
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
easy_install pybrain
pip install Pyro4
pip install --upgrade easygui
