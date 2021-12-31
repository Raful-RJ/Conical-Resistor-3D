from Laplace3D import *

def fun(x):
    f = 1 - 0.5*x
    df = -0.5
    return (f,df)

c = Laplace3D(0.05,1,sigma = 1,f = fun)
c.compile_pot()

cuts1 = [[-0.7,-0.5,-0.2],[0,0.4,0.8]]
cuts2 = [[0.1,0.2,0.4],[0.5,0.8,1]]
names = ['plot2d_1','plot2d_2']
[c.plot2d(names[i],cuts1[i],cuts2[i]) for i in range(len(names))]

        
        

    
        
        
        

    
