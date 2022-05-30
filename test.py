# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:35:29 2022

@author: fwillsmore
"""

import numpy as np

#%% Knapsack problem

# As a first example, consider the solution of the 0/1 knapsack problem: given a set of I items, each one with weight wi and estimated profit pi, one wants to select a subset with maximum profit such that the summation of the weights of the selected items is less or equal to the knapsack capacity c. Considering a set of decision binary variables xi that receive value 1 if the i-th item is selected, or 0 if not, we solve the following:

from mip import Model, xsum, maximize, BINARY

p = [10, 13, 18, 31, 7, 15] # profit for each variable x
w = [11, 15, 20, 35, 10, 33] # weight of each variable x
c, I = 45, range(len(w)) # total weight capacity c, and the index of items 0,1,...,5 in x

m = Model("knapsack") # create MIP model called "knapsack" 

x = [m.add_var(var_type=BINARY) for i in I] # add binary decision variables

m.objective = maximize(xsum(p[i] * x[i] for i in I)) # maximise the profit

m += xsum(w[i] * x[i] for i in I) <= c # add constraint total weight <= capacity

m.optimize() # solve the model

selected = [i for i in I if x[i].x >= 0.99]
print("selected items: {}".format(selected))

profit = sum(np.array(p)[selected])
print("total profit is: {}".format(m.objective_value))

weight = sum(np.array(w)[selected])
print("total weight is: {}".format(weight))

#%% Travelling Salesmen Problem

# To illustrate this problem, consider that you will spend some time in Belgium and wish to visit some of its main tourist attractions. You want to find the shortest possible tour to visit all these places. More formally considering V = {0,...,n-1} and a distance matrix D (nxn) with elements cij in positive real numbers, a solution consists of exactly n (origin, destination) pairs indicating the itinery of your trip.

from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

# names of places to visit
places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
          'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
          'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
          'Remouchamps', 'Waterloo']

# distances in an upper triangular matrix
dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
         [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
         [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
         [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
         [51, 114, 72, 54, 69, 139, 105, 155, 62],
         [70, 25, 22, 52, 90, 56, 105, 16],
         [45, 61, 111, 36, 61, 57, 70],
         [23, 71, 67, 48, 85, 29],
         [74, 89, 69, 107, 36],
         [117, 65, 125, 43],
         [54, 22, 84],
         [60, 44],
         [97],
         []]

# number of nodes and list of vertices
n, V = len(dists), set(range(len(dists)))

# distances matrix
c = [[0 if i == j
      else dists[i][j-i-1] if j > i
      else dists[j][i-j-1]
      for j in V] for i in V]

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

# continuous variable to prevent subtours: each city will have a
# different sequential id in the planned route except the first one
y = [model.add_var() for i in V]

# objective function: minimize the distance
model.objective = minimize(xsum(c[i][j]*x[i][j] for i in V for j in V))

# constraint : leave each city only once
for i in V:
    model += xsum(x[i][j] for j in V - {i}) == 1

# constraint : enter each city only once
for i in V:
    model += xsum(x[j][i] for j in V - {i}) == 1

# subtour elimination
for (i, j) in product(V - {0}, V - {0}):
    if i != j:
        model += y[i] - (n+1)*x[i][j] >= y[j]-n

# optimizing
model.optimize()

# checking if a solution was found
if model.num_solutions:
    out.write('route with total distance %g found: %s'
              % (model.objective_value, places[0]))
    nc = 0
    while True:
        nc = [i for i in V if x[nc][i].x >= 0.99][0]
        out.write(' -> %s' % places[nc])
        if nc == 0:
            break
    out.write('\n')