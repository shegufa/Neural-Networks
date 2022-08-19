import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sig_der(x):
    return x*(1-x)

data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

input_layer = data[:,0:2]
desired_output = data[:,2]
desired_output = np.array([desired_output]).T

v = np.random.random((2,2))
w = np.random.random((2,1))
v = np.array([[0.59460266,0.36001941], [0.72998028, 0.83658554]])
w = np.array([[0.15650103],[0.56883396]])

number_of_epochs = 10

n = 2.3
while(n<2.5):
    print("n ",str(n))
    for epoch in range(number_of_epochs):
        hidden_layer = sigmoid(np.dot(input_layer,v)) #(4,2)
       
        output_layer = sigmoid(np.dot(hidden_layer,w)) #(4,2) * (2*1) = (4*1)
        #print(output_layer.shape)
        ol_error = desired_output - output_layer #(4,1)
        #print(ol_error.shape)
        ol_delta = sig_der(output_layer) * ol_error    #(4,1)
        #print(ol_delta.shape)
        hl_error = np.dot(ol_delta,w.T) #(4,1) * (1*2) = (4*2)
        hl_delta = sig_der(hidden_layer) * hl_error #(4,2)
        
        w+= n*np.dot(hidden_layer.T, ol_delta) #(2*4) * (4*1) = (2*1)
        v+= n*np.dot(input_layer.T, hl_delta) #(2*4) * (4*2) = (2*2)
        
        print("Epoch ",epoch)
        print(output_layer)
        print(v)
        print(w)

    n = round(n+0.3,2)