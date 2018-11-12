import numpy as np
from crf import CRF

np.random.seed(2018)

def verify_gradients():
 
    model = CRF(input_size=10, classes=3)
    example = (np.random.rand(4,10),np.array([0,1,1,2]))
    input, target = example
    epsilon=1e-6

    model.w = [0.01*rng.rand(model.input_size,model.classes),
                       0.01*rng.rand(model.input_size,model.classes),
                        0.01*rng.rand(model.input_size,model.classes)]
    model.b = 0.01*rng.rand(model.classes)
    model.w_edge = 0.01*rng.rand(model.classes,model.classes)
        
    model.forward(input,target)
    model.backward(input,target) 

    import copy
    emp_dw = copy.deepcopy(model.w)
  
    for h in range(len(model.w)):
        for i in range(model.w[h].shape[0]):
            for j in range(model.w[h].shape[1]):
                model.w[h][i,j] += epsilon
                a = model.forward(input,target)
                model.w[h][i,j] -= epsilon
               
                model.w[h][i,j] -= epsilon
                b = model.forward(input,target)
                model.w[h][i,j] += epsilon
                
                emp_dw[h][i,j] = (a-b)/(2.*epsilon)


    print 'dw[-1] diff.:',np.sum(np.abs(model.dw[-1].ravel()-emp_dw[-1].ravel()))/model.w[-1].ravel().shape[0]
    print 'dw[0] diff.:',np.sum(np.abs(model.dw[0].ravel()-emp_dw[0].ravel()))/model.w[0].ravel().shape[0]
    print 'dw[1] diff.:',np.sum(np.abs(model.dw[1].ravel()-emp_dw[1].ravel()))/model.w[1].ravel().shape[0]
  
    emp_dw_edge = copy.deepcopy(model.w_edge)
  
    for i in range(model.w_edge.shape[0]):
        for j in range(model.w_edge.shape[1]):
            model.w_edge[i,j] += epsilon
            a = model.forward(input,target)
            model.w_edge[i,j] -= epsilon
            model.w_edge[i,j] -= epsilon
            b = model.forward(input,target)
            model.w_edge[i,j] += epsilon
                
            emp_dw_edge[i,j] = (a-b)/(2.*epsilon)


    print 'dw_edge  diff.:',np.sum(np.abs(model.dw_edge.ravel()-emp_dw_edge.ravel()))/model.w_edge.ravel().shape[0]

    emp_db = copy.deepcopy(model.b)
        for i in range(model.b.shape[0]):
            model.b[i] += epsilon
            a = model.forward(input,target)
            model.b[i] -= epsilon
            
            model.b[i] -= epsilon
            b = model.forward(input,target)
            model.b[i] += epsilon
            
            emp_db[i] = (a-b)/(2.*epsilon)

        print 'db diff.:',np.sum(np.abs(model.db.ravel()-emp_db.ravel()))/model.db.ravel().shape[0]


if __name__ == "__main__":
    verify_gradients()
