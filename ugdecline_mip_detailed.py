# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:58:52 2022

@author: fwillsmore
"""

import pandas as pd
import numpy as np
from mip import Model, minimize, OptimizationStatus, xsum, BINARY
from plotnine import *
from plotnine.data import mpg

# !! need to sort out timing and units

## Time
# discretised time interval size (mins)
tint = 5
# time horizon (hours)
Tend = 3
# array of time points
t = 1/60*np.array(range(0,60*Tend,tint))
start = t[:-1] # start time of interval t for all t in T
end = t[1:] # end time of interval t for all t in T
# set of time intervals
T = range(len(t)-1)

## Decline
# !! possibility to extend to uneven segments to reduce computations - however concerns around constraints failing
# ramp length (km)
X_dist_m = [0,
          201.3 + 423.8,
          568.8 + 53.1,
          132.8 + 39.3,
          183,
          142,
          31,
          130.1 + 54.5,
          131.4 + 43,
          133.8 + 51.5]
X_dist = [x/1000 for x in X_dist_m]
# number of segments
Xseg = len(X_dist)
# passing bay constraint
# we assume passing bays are pre-determined
z = [0,1,1,1,0,1,0,1,1,1,0]

## Vehicles
# !! possibility to extend to different loading and unloading times for each vehicle
# number of vehicles
V = set(range(2))
# vehicle velocities in the down direction
vd = [20,15,25]
# vehicle velocities in the up direction
vu = [15,10,5]
# loading time (hours)
tl = 10/60
# unloading time (hours)
tu = 10/60

## Schedule
# two loading stations one early and one at bottom
# no. of trips is 10, three going to the first and the rest to the last
# total number of trips
# numTrips = 10
 # set of all trips
# J = range(numTrips)
# set of trips for each vehicle
# Jv = [[0,3,6,9],[1,4,7],[2,5,8]]
# destination of each trip / loading point
# s = [2,5,5,2,5,5,5,5,5,2]

# !! testing
J = range(5)
Jv = [[0,1,2],[3,4]]
s = [2,5,2,5,9]

# set of segments down to the loading point for each trip
L = [set(range(sj+1+1)) for sj in s]

# define the milp
model = Model("ugdecline")

# departure time 
# D(jl)^d/u = departure time in the down/up direction by trip j from segment l in Lj\{sj+1}
# !! have included variables that will not be used but maintain indexing
Dd = [[model.add_var('Dd({},{})'.format(j,l)) for l in L[j]] for j in J]
Du = [[model.add_var('Du({},{})'.format(j,l)) for l in L[j]] for j in J]
# the variable D(j0)^u is not used since it does not make sense leaving segment 0 (surface) in the up direction

# arrival time
# A(jl)^d/u = arrival time in the down/up direction by trip j to segment l in Lj\{0}
Ad = [[model.add_var('Ad({},{})'.format(j,l)) for l in L[j]] for j in J]
Au = [[model.add_var('Au({},{})'.format(j,l)) for l in L[j]] for j in J]

# add the objective function variable since we don't know which vehicle makes the last trip
objf = model.add_var('Obj')
# add the constraint to calculate objf
for j in J:
    model += objf >= Au[j][0] # equal to the arrival time of the last trip

# minimise final trip time + unloading
model.objective = minimize(objf + tu)
# a = 0.001
# model.objective = minimize(objf + tu + a*xsum(Dd[j][l-1]-Ad[j][l] for l in L[j]-{0,s[j]+1} for j in J))

# sets the ready time for the first trip of every vehicle at the surface of the mine
# !! could be extendend such that different vehicles have different starting points Ajl == 0 for 
for v in V:
        model += Ad[Jv[v][0]][0] == 0

# calculate the arrival time of a segment from the depature time of the previous one.    
for v in V:
    for j in Jv[v]:
        for l in L[j]:
            if l not in {0,s[j]+1}:
                model += Ad[j][l+1] == Dd[j][l-1]+X_dist[l]/vd[v]
            if l != s[j]+1:
                model += Au[j][l-1] == Du[j][l+1]+X_dist[l]/vu[v]

# if a passing bay exists between two segments, then the departure time may occur later than the arrival time
M = 10^6
for v in V:
    for j in Jv[v]:
        for l in L[j]-{0,s[j]+1}:
            model += Dd[j][l-1]+M*(1-z[l]) >= Ad[j][l]
            model += Dd[j][l-1]-M*z[l] <= Ad[j][l]
            model += Dd[j][l-1] >= Ad[j][l]-M*z[l] 

# only one vehicle can be loaded at a time, which is why vehicles may form a queue at the passing bay right before a loading station segment. More thoroughly handled during collision.
for v in V:
    for j in Jv[v]:
        model += Dd[j][s[j]] >= Ad[j][s[j]+1]
     
# after loading, the truck will have its direction changed from down to up.
for v in V:
    for j in Jv[v]:
        model += Au[j][s[j]] == Dd[j][s[j]]+tl

# vehicles may also wait at the loading station after they have been loaded.
for v in V:
    for j in Jv[v]:
        model += Du[j][s[j]+1] >= Au[j][s[j]] # !!

# loaded trucks cannot stop while going up
for v in V:
    for j in Jv[v]:
        for l in L[j]-{0,s[j]+1}:
            model += Du[j][l] == Au[j][l-1]
            
# vehicle has unloaded and is ready to go back down the ramp
for v in V:
    for j in Jv[v]: 
        if j != Jv[v][0]:
            model += Ad[j][0] == Au[j-1][0]+tu
            
# !! temporary to fix trip linking            
for v in V:
    for j in Jv[v]: 
        model += Dd[j][0] >= Ad[j][0]
            
# binary variable to locate truck positions on segments
xd = [[[model.add_var('xd({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]
xu = [[[model.add_var('xu({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]
# auxillary variables
WDdx = [[[model.add_var('WDdx({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]
WAdx = [[[model.add_var('WAdx({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]
WDux = [[[model.add_var('WDux({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]
WAux = [[[model.add_var('WAux({},{},{})'.format(j,l,t),var_type=BINARY) for t in T] for l in L[j]] for j in J]

eps = 10 ** (-6)

# set xd[j][l][t] = 1 iff WAdx & WDdx = 1
for v in V:
    for j in Jv[v]:
        for l in L[j]-{0}:
            for t in T:
                model += Dd[j][l-1] >= end[t] + eps - M*WDdx[j][l][t]
                model += Dd[j][l-1] <= end[t] + M*(1-WDdx[j][l][t])
                if l != s[j] + 1:
                    model += Ad[j][l+1] <= start[t]-eps + M*WAdx[j][l][t]
                    model += Ad[j][l+1] >= start[t] - M*(1-WAdx[j][l][t])
                model += WDdx[j][l][t] >= xd[j][l][t]
                model += WAdx[j][l][t] >= xd[j][l][t]
                model += WAdx[j][l][t] + WDdx[j][l][t] <= xd[j][l][t] + 1
                
# set xu[j][l][t] = 1 iff WAux & WDux = 1
for v in V:
    for j in Jv[v]:
        for l in L[j]-{0}:
            for t in T:
                if l != s[j] + 1:
                    model += Du[j][l+1] >= end[t] + eps - M*WDux[j][l][t]
                    model += Du[j][l+1] <= end[t] + M*(1-WDux[j][l][t])
                model += Au[j][l-1] <= start[t]-eps + M*WAux[j][l][t]
                model += Au[j][l-1] >= start[t] - M*(1-WAdx[j][l][t])
                model += WDux[j][l][t] >= xu[j][l][t]
                model += WAux[j][l][t] >= xu[j][l][t]
                model += WAux[j][l][t] + WDux[j][l][t] <= xu[j][l][t] + 1

# model occupied segments to avoid collisions
for v in V:
    for j in Jv[v]:
        for l in L[j]-{0}:
            for t in T:
                model += 1-xu[j][l][t] >= xsum(xd[j_][l][t]+xu[j_][l][t] for j_ in J if j_ not in Jv[v] if l <= s[j_])
                

## Print Optimal Value and Solution
status = model.optimize(max_seconds=300)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('solution:')
    for v in model.vars:
       # if abs(v.x) > 1e-6: # only printing non-zeros
       print('{} : {}'.format(v.name, v.x))

## Visualise Solution

sol_ls = []

for j in J:
    for l in L[j]:
        # vehicle_id = np.where([j in v for v in Jv])
        vehicle_id = [i for i,val in enumerate([j in v for v in Jv]) if val][0]+1
        if l == s[j]+1: # !! copying some stuff better way to code
            l = s[j]
            sol_ls.append(['Du({},{})'.format(j,l),vehicle_id,j,l,Du[j][l+1].x])
            sol_ls.append(['Ad({},{})'.format(j,l),vehicle_id,j,l,Ad[j][l+1].x])
        sol_ls.append(['Dd({},{})'.format(j,l),vehicle_id,j,l,Dd[j][l].x])
        if l != 0:
            sol_ls.append(['Ad({},{})'.format(j,l),vehicle_id,j,l-1,Ad[j][l].x])
            sol_ls.append(['Du({},{})'.format(j,l),vehicle_id,j,l-1,Du[j][l].x])
        sol_ls.append(['Au({},{})'.format(j,l),vehicle_id,j,l,Au[j][l].x])
sol_df = pd.DataFrame(sol_ls,columns = ['variable','id','trip','segment','time'])
sol_df['id'] = sol_df['id'].map(str)

sol_df['depth'] = np.cumsum(X_dist)[sol_df['segment']]

segments = pd.DataFrame({'segment': range(10),'depth': np.cumsum(X_dist)})

(ggplot(sol_df) +
 aes(x = 'time',y = 'depth',group = 'id') +
 geom_line(aes(color = 'id')) +
 geom_hline(aes(yintercept = 'depth'),linetype = 'dotted',data = segments,alpha = 0.5) +
 scale_y_reverse(breaks = segments['depth']) +
 scale_color_discrete(name = 'Vehicle') +
 theme_classic() +
 labs(x = "Time (h)", y = "Depth (km)")
 )

print('model has {} vars, {} constraints and {} nzs'.format(model.num_cols, model.num_rows, model.num_nz))

# at time interval 19 vehicle 1 should wait in passing bay for vehicle 2 to pass
j = 1
l = 6
Du[j][l].x
Au[j][l-1].x
