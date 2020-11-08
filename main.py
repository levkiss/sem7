import logging
from math import sqrt
import numpy as np
from scipy.optimize import linprog


class Vector2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector2D({}, {})'.format(self.x, self.y)

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    # def __sub__(self, number):
    #   return Vector2D(self.x - number, self.y - number)

    def __mul__(self, number):
        return Vector2D(self.x * number, self.y * number)

    def mul_num(self, number):
        return Vector2D(self.x * number, self.y * number)

    def scalar_mult(self, other):
        return self.x * other.x + self.y * other.y

    def __neg__(self):
        return Vector2D(-self.x, -self.y)

    def normalize(self):
        return sqrt((self.x) ** 2 + (self.y) ** 2)

    def normalize_sqr(self):
        return self.x * self.y + self.y * self.y


class linearization:
    def __init__(self, x=5.0, y=5.0):
        self.u_old = np.array([x, y])

        self.BETTA = 0.000001
        self.eps = 0.00001

        # заданные начальные ограничения типа неравенство
        g1 = lambda u: -u[1] - 1
        g2 = lambda u: u[0] + 1

        self.restrictions = [g1, g2]

        # целевая функция x1^2+ x2^2 --> min
        self.f = lambda x: (x[0] - 1)**2 + (x[1] + 1)**2

        #self.w_k = lambda u: self.g1(self.u_old) + self.gradient_g1(self.u_old).scalar_mult(u - self.u_old)

    def run_count(self):
        # find u1
        iter = 0
        print('The count is starting')
        # grad = self.gradient(self.f, self.u_old)
        #print('W_k_0 = ', self.w_k(self.u_old))
        u_k = self.u_old

        self.u_old = self.u_old + 10
        print("Номер итерации - точка - значение функции")
        print(0, u_k, self.f(u_k))
        alpha = 10
        while (abs(self.f(u_k) - self.f(self.u_old)) > self.eps) and iter < 50:
            grad = self.gradient(self.f, u_k)
            #print('grad = ', grad)
            gradients_resrt = np.array([self.gradient(g, u_k) for g in self.restrictions])
            delta = (gradients_resrt * u_k).sum(axis=1) - np.array([g(u_k) for g in self.restrictions])
            #print('a = ', A_ub, ' b = ', B_ub)
            bounds = np.c_[u_k - alpha, u_k + alpha]
            #print('bounds = ', bounds)
            self.u_old = u_k
            res = linprog(grad, gradients_resrt, delta, bounds=bounds, method="simplex")
            u_k = res.x
            alpha = alpha/2
            iter += 1
            print(iter, "%.4f" % u_k[0], "%.4f" % u_k[1], "%.4f" % self.f(u_k))
        print('The answer gives function value about', self.f(u_k))

    def gradient(self, f, u):
        #f_grad = np.array([0.0, 0.0])
        # x_x = [x.x, x.y]
        u_0 = u.reshape(-1)
        f_grad = np.zeros_like(u)
        # print('fgrad = ', f_grad)
        for i in range(u.shape[0]):
            # print('x_o.shape = ', x_0.shape[0])
            x_plus = u_0.copy()
            x_plus[i] += self.eps
            x_minus = u_0.copy()
            x_minus[i] -= self.eps

            # x_pl = Vector2D(x_plus[0], x_plus[1])
            # x_mn = Vector2D(x_minus[0], x_minus[1])
            f_grad[i] = f(x_plus) - f(x_minus)
            f_grad[i] /= 2 * self.eps
        return f_grad

    def run_method(self):
        print('For more info about program press 1')
        while 1:
            a = input('>>')
            if a == '1':
                print(
                    'You can do the following instructions:\n1 - press for help\n2 - press to change start point (enter first coordinate, then press enter)\n3 - to change epsilon(please enter with dot)\n4 - to change betta\n5 - to start counting\n6 - to print function\n7 - to exit')
            elif a == '2':
                x = int(input('>>'))
                y = int(input('>>'))
                self.u_old = np.array([x, y])
            elif a == '3':
                self.eps = float(input('>>'))
            elif a == '4':
                self.BETTA = float(input('>>'))
            elif a == '5':
                self.run_count()
            elif a == '6':
                print('J(u) = x1^2+ x2^2 --> min')
            elif a == '7':
                exit()
            else:
                print('Wrong command, please try again')



def argmin_u(self, func, u_k):
    start = Vector2D(self.x_min, self.y_min)
    j_max = func(start, u_k)
    # print(j_max)
    point_min = u_k
    # print(u_k)
    for i in np.arange(self.x_min, self.x_max, self.eps):
        for j in np.arange(self.y_min, self.y_max, self.eps):
            vec2 = Vector2D(i, j)
            # print(vec2)
            res = func(vec2, u_k)
            # print(res)
            if res < j_max:
                j_max = res
                point_min = vec2
                # print(point_min)
                # print(j_max)
    return point_min


def argmin_alpha(self, func, u_k, u_k_):
    alp = 0
    j_max = func(alp, u_k, u_k_)
    # print(j_max)
    for i in np.arange(0, 1, self.BETTA):
        # print(vec2)
        res = func(i, u_k, u_k_)
        # print(res)
        if res < j_max:
            j_max = res
            alp = i
    return alp


def grad_method(self, u_n):
    # print(u_n.x)
    J_k_1 = lambda u, u_k: 2 * u_k.x * (u.x - u_k.x) + 2 * u_k.y * (u.y - u_k.y)
    J_k_2 = lambda alpha, u_k, u_k_: (u_k.x + alpha * (u_k_.x - u_k.x)) ** 2 + (
            u_k.y + alpha * (u_k_.y - u_k.y)) ** 2

    Phi_k = lambda u, u_k: 0.5 * ((u.x - u_k.x) ** 2 + (u.y - u_k.y) ** 2) + self.BETTA * (
            2 * u_k.x * (u.x - u_k.x) + 2 * u_k.y * (u.y - u_k.y))

    J_k_u = lambda u, u_k: (u.x - u_k.x + self.BETTA * 2 * u_k.x) * (u.x - u_k.x) + (
            u.y - u_k.y + self.BETTA * 2 * u_k.y) * (u.y - u_k.y)
    J_k_alpha = lambda alpha, u_k, u_k_: 0.5 * (
            (u_k.x + alpha * (u_k_.x - u_k.x)) ** 2 + (u_k.y + alpha * (u_k_.y - u_k.y)) ** 2) + self.BETTA * (
                                                 2 * u_k.x * (u_k.x + alpha * (u_k_.x - u_k.x)) + 2 * u_k.y * (
                                                 u_k.y + alpha * (u_k_.y - u_k.y)))

    u__n = self.argmin_u(J_k_1, u_n)

    while u__n.__sub__(u_n).normalize() > self.eps:
        print('iteration of grad method')
        u__n = self.argmin_u(J_k_u, u_n)
        alpha_k = self.argmin_alpha(J_k_alpha, u_n, u__n)
        u_n = u_n.__add__((u__n.__sub__(u_n)).__mul__(alpha_k))
        print(u_n)
    return u_n


def gradient_phi(self, u, u_old):
    grad = self.gradient_f(u_old).mul_num(self.BETTA) + (u - u_old)
    return grad


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    li = linearization()
    # first = Vector2D(-1, 1)
    # print(li.gradient(first))
    li.run_method()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
