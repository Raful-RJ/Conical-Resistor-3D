__all__ = ['mesh3D']

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def mesh3D(h,H1, f):

    def generate_pid(i,j,k):
        return str(i) + '.' + str(j) + '.' + str(k)

    def F(f,X):
        Y = []
        for x in X:
            Y.append(f(x)[0])
        return np.array(Y)

    def raio(x,y):
        return (x**2 + y**2)**0.5
    r0 = f(H1)
    r1 = f(H1)
    H = np.arange(0,H1 + h,h)
    R = F(f,H)
    Rmax = max(R)

    lim_fun = np.array([H,R])

    X,Y,Z = np.mgrid[-Rmax:Rmax+h:h,-Rmax:Rmax+h:h,0:H1+h:h]
    coord = []
    for k in range(X.shape[2]):
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                if raio(X[i][j][k],Y[i][j][k]) <= f(Z[i][j][k])[0]:
                    pid = generate_pid(i,j,k)
                    
                    #test for front neighbor (x axis)
                    try:
                        if raio(X[i][j+1][k],Y[i][j+1][k]) <= f(Z[i][j+1][k])[0]:
                            front = generate_pid(i,j+1,k)
                        else: front = '-1'
                    except: front = '-1'
                    
                    #test for back neighbor (x axis)
                    try:
                        if raio(X[i][j-1][k],Y[i][j-1][k]) <= f(Z[i][j-1][k])[0] and (j-1)>=0:
                            back = generate_pid(i,j-1,k)
                        else: back = '-1'
                    except: back = '-1'
                            
                    #test for left neighbor (y axis)
                    try:
                        if raio(X[i-1][j][k],Y[i-1][j][k]) <= f(Z[i-1][j][k])[0]and (i-1)>=0:
                            left = generate_pid(i-1,j,k)
                        else: left = '-1'
                    except: left = '-1'
                    #test for right neighbor
                    try:
                        if raio(X[i+1][j][k],Y[i+1][j][k]) <= f(Z[i+1][j][k])[0]:
                            right = generate_pid(i+1,j,k)
                        else: right = '-1'
                    except: right = '-1'

                    #test for up neighbor (z axis)
                    try:
                        if raio(X[i][j][k+1],Y[i][j][k+1]) <= f(Z[i][j][k+1])[0]:
                            up = generate_pid(i,j,k+1)
                        else: up = '-1'
                    except: up = '-1'
                    #test for down neighbor (z axis)
                    try:
                        if raio(X[i][j][k-1],Y[i][j][k-1]) <= f(Z[i][j][k-1])[0] and (k-1)>=0:
                            down = generate_pid(i,j,k-1)
                        else: down = '-1'
                    except: down = '-1'

                    #setting boundary values
                    if Z[i][j][k] == 0: #bottom
                        potential_value = 10 
                    elif Z[i][j][k] == H1: #top
                        potential_value = 0
                    elif '-1' in [front,back,left,right,up,down]: #sides
                        potential_value = 1
                    else: potential_value = np.nan
                    
                    coord.append([pid,X[i][j][k],Y[i][j][k],Z[i][j][k],potential_value,0,0,0,front,back,left,right,up,down])

    general_data = np.array([r0,r1,h])
     
    features = np.array(['Id','x','y','z','potential','Jx','Jy','Jz','front_neighbor','back_neighbor','left_neighbor','right_neighbor','up_neighbor','down_neighbor'])
    coord = np.reshape(np.array(coord, dtype = 'object'),(len(coord),features.shape[0]))
    
    return X,Y,Z,lim_fun,features,coord, general_data
  




            
