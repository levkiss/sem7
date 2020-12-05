import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='logs.log', format='%(message)s', filemode='w', level=logging.DEBUG)



x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
u_old = np.array(x0)
u_k = np.array(x0)

BETTA = 0.000001
eps = 0.00001
alpha = 10

def f(x):
    sum_1 = 0
    for i in range(1, 5):
        sum_1 += (x[i] - (i + 1) * x[0]) ** 4
    sum_1 *= 150
    return sum_1 + (x[0] - 1) ** 2

def g1(x):
    sum_1 = 0
    for i in range(5):
        sum_1 += ((i+1) * x[i])**2
    return sum_1 - 224

def phi_k(u, u_k):
    first_num = 0
    sec_num = 0
    u_sub = u - u_k
    for i in range(5):
        first_num += u_sub[i] * 2
    first_num = first_num * 0.5
    gradi = gradient(f, u_k)
    for i in range(5):
        sec_num = sec_num + gradi[i] * u_sub[i]
    sec_num *= BETTA
    return first_num + sec_num

def phi_k2(u, u_k):
    thiss = 0
    fi = u - (u_k - BETTA*gradient(f, u_k))
    for i in range(5):
        thiss += fi[i]**2
    return thiss*0.5

def gradient(f, u):
        #f_grad = np.array([0.0, 0.0])
        # x_x = [x.x, x.y]
    u_0 = u.reshape(-1)
    f_grad = np.zeros_like(u)
        # print('fgrad = ', f_grad)
    for i in range(u.shape[0]):
        # print('x_o.shape = ', x_0.shape[0])
        x_plus = u_0.copy()
        x_plus[i] += eps
        x_minus = u_0.copy()
        x_minus[i] -= eps

        f_grad[i] = f(x_plus) - f(x_minus)
        f_grad[i] /= 2 * eps
    return f_grad

def cons_f(x):
     return [x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + 4*x[3]**2 + 5*x[4]**2]

def cons_J(x):
     return [[2*x[0], 4*x[1], 6*x[2], 8*x[3], 10*x[4]]]

def cons_H(x, v):
     return v[0]*np.array([[2, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 6, 0, 0], [0, 0, 0, 8, 0], [0, 0, 0, 0, 10]])

def run():
    iter = 0
    global u_k

    print('The count is starting')
    # grad = self.gradient(self.f, self.u_old)
    # print('W_k_0 = ', self.w_k(self.u_old))
    u_old = np.array(x0)
    #u_k = u_old

    u_old = u_old + 10
    logging.info('Iter number - point - function value')
    logging.info('0 - %s - %s', u_k, f(u_k))
    alpha = 10

    #plt.plot(f(u_k), iter, 'red')
    while (abs(f(u_k) - f(u_old)) > eps) and (iter < 150):
        u_old = u_k
        nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 224, jac=cons_J, hess=cons_H)
        #bounds = np.c_[u_k - alpha, u_k + alpha]
        res = minimize(phi_k2, u_k, method='trust-constr', args=(u_k), constraints=[nonlinear_constraint],
                       options={'xtol': 1e-3, 'disp': False})
        u_k = res.x
        # alpha = alpha/2

        plt.plot(f(u_k), iter, 'bo')
        iter += 1
        iter_print = ["%.2f" % u_k[i] for i in range(5)]
        logging.info('%s - %s - %s', iter, iter_print, "%.4f" % f(u_k))
    print('The answer gives function value about', f(u_k))
    print('Cons function gives ', cons_f(u_k))
    plt.xlabel('function value')
    plt.ylabel('iteration number')
    plt.title('Linearization method')
    plt.show()

def start():
    print('For more info about program press 1')
    while 1:
        a = input('>>')
        if a == '1':
            print(
                'You can do the following instructions:\n1 - press for help\n2 - press to change start point (enter first coordinate, then press enter, ...)\n3 - to change epsilon(please enter with dot)\n4 - to start counting\n5 - to exit')
        elif a == '2':
            global x0
            x1 = int(input('>>>>'))
            x2 = int(input('>>>>'))
            x3 = int(input('>>>>'))
            x4 = int(input('>>>>'))
            x5 = int(input('>>>>'))
            x0 = np.array([x1, x2, x3, x4, x5], dtype=float)
        elif a == '3':
            global eps
            eps = float(input('>>'))
        elif a == '4':
            run()
        elif a == '5':
            exit()
        else:
            print('Wrong command, please try again')

start()
'''
print(gradient(f, u_k))
print(phi_k2(u_k, u_k+0.1))
bounds = np.c_[u_k - alpha, u_k + alpha]

res = minimize(phi_k2, u_k, method='POWELL', args=(u_k), bounds=bounds,
               options={'xtol': 1e-15, 'disp': True})
print(res.x)
print(phi_k2(res.x, u_k))
print(f(u_k))

u_k = res.x
bounds = np.c_[u_k - alpha, u_k + alpha]
res = minimize(phi_k2, u_k, method='POWELL', args=(u_k), bounds=bounds,
               options={'xtol': 1e-15, 'disp': True})
print(res.x)
print(phi_k2(res.x, u_k))
print(f(u_k))
'''