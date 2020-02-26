import matplotlib.pyplot as plt
import numpy as np

class loss_fn(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def forward(self):
        loss_x = 0.5*self.x**2 + 10*self.x*self.y
        loss_y = 0.5*self.y**2 - 10*self.x*self.y
        return (loss_x, loss_y)
    def backward(self):
        dldx = self.x + 10*self.y
        dldy = self.y - 10*self.x
        return (dldx, dldy)

def sgd(lr, lf):
    lf.x += -lr * lf.backward()[0]
    lf.y += -lr * lf.backward()[1]

def get_sym_adj():
    d2ldx2 = 1
    d2ldxdy = 10
    d2ldydx = -10
    d2ldy2 = 1
    h = np.matrix([[d2ldx2, d2ldxdy], [d2ldydx, d2ldy2]])
    h_t = h.transpose()
    a_t = (h_t-h)/2
    return a_t

def sga(lr, lf):
    grad = np.asmatrix(lf.backward()).transpose()
    adjusted_grad = grad + get_sym_adj()*grad
    #print(adjusted_grad)
    lf.x += -lr * adjusted_grad[0,0]
    lf.y += -lr * adjusted_grad[1,0]


def test1(lr, pos):
    x, y = 10, 10
    lf = loss_fn(x, y)
    xs = list()
    ys = list()

    for _ in range(100):  
        sgd(lr, lf)
        print('x:{}, y:{}, loss:{}'.format(lf.x, lf.y, lf.forward()))
        #draw_point(lf.x,lf.y)
        xs.append(lf.x)
        ys.append(lf.y)

    plt.subplot(2,3,pos)
    plt.scatter(xs,ys)
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(x * np.cos(theta), x * np.sin(theta))
    plt.axis("equal")
    plt.axis([-2*x, 2*x, -2*x, 2*x])
    return (xs,ys)

def test2(lr, pos):
    x, y = 10, 10
    lf = loss_fn(x, y)
    xs = list()
    ys = list()
    for _ in range(100):
        sga(lr, lf)
        print('x:{}, y:{}, loss:{}'.format(lf.x, lf.y, lf.forward()))
        xs.append(lf.x)
        ys.append(lf.y)

    plt.subplot(2,3,pos)
    plt.scatter(xs,ys)
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(x * np.cos(theta), x * np.sin(theta))
    plt.axis("equal")
    plt.axis([-2*x, 2*x, -2*x, 2*x])
    return (xs,ys)

if __name__ == "__main__":
    lrs = (0.001, 0.005, 0.01)
    for i in range(3):
        test1(lrs[i], i+1)
        test2(lrs[i], i+4)
    plt.show()

