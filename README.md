# Quantum computing and Classical computing 
In this repository you will find tutorials and Examples/Algorithm related to Qunatom computing and Classical computing.I try to make the code as clear as possible, and the goal is be to used as a learning resource and a way to lookup problems to solve specific problems. We have discussed about two jupyter notebooks and these two jupyter notebook contains scikit-learn notebook and quantum notebook

#### Approch:
##### Classical computers manipulate ones and zeroes to crunch through operations, but quantum computers use quantum bits or qubits. Just like classical computers, quantum computers use ones and zeros, but qubits have a third state called “superposition” that allows them to represent a one or a zero at the same time. Instead of analysing a one or a zero sequentially, superposition allows two qubits in superposition to represent four scenarios at the same time. Therefore, the time it takes to crunch a data set is significantly reduced.

## Scikit-learn:
Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

In the 'scikit-learn.ipynb' notebook I have define servel algorithm of classical algorithm (Scikit-learn Algorithm).

## Deutsch’s algorithm:
Deutsch-Jozsa algorithm is one of the first quantum algorithms with nice speedup over its classical counterpart. It showed that there can be advantages to using a quantum computer as a computational tool for a specific problem.

#### Deutsch-Jozsa Problem 
We are given a hidden Boolean function f, which takes as input a string of bits, and returns either 0 or 1, that is:

f({x0,x1,x2,...})→0 or 1 , where xn is 0 or 1
 
The property of the given Boolean function is that it is guaranteed to either be balanced or constant. A constant function returns all 0 's or all  1 's for any input, while a balanced function returns 0 's for exactly half of all inputs and 1 's for the other half. Our task is to determine whether the given function is balanced or constant.

Note that the Deutsch-Jozsa problem is an n -bit extension of the single bit Deutsch problem.

In 'quantum-deutsch.ipynb' jupyter notebook I defined Deutsch-Jozsa Problem's solution in which we can solve this problem with 100% confidence using Quantom computing solution.

# How to Run Notebook Files:
You can simply clone this reposetry and run these files in jupyter notebook. Before Running these files you have to install all the required libraries.

OR

You can run these files using Docker.

Install Docker in your system : 'https://www.docker.com/get-started'

In your terminal run docker : '$ docker'

First you have to run requirements.txt file for required libraries :
"!pip install -r requirements.txt" or using Docker image "sudo docker build -t assess -f Dockerfile ."

Run Jupyter Docker '$ sudo docker run -it -p 8888:8888 assess'. Click any link of Docker server.
