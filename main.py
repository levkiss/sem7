import logging
from math import sqrt

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

    #def __sub__(self, number):
     #   return Vector2D(self.x - number, self.y - number)

    def __mul__(self, number):
        return Vector2D(self.x*number, self.y*number)

    def mul_num(self, number):
        return Vector2D(self.x * number, self.y * number)

    def scalar_mult(self, other):
        return self.x * other.x + self.y * other.y

    def __neg__(self):
        return Vector2D(-self.x, -self.y)

    def normalize(self):
        return sqrt((self.x)**2 + (self.y)**2)

    def normalize_sqr(self):
        return self.x * self.y + self.y * self.y

class linearization:
    def __init__(self, x=1, y=2):
        self.u_old = Vector2D(x, y)

        self.BETTA = 0.0001
        self.eps = 0.1

        self.w_k = lambda u: self.g1(self.u_old) + self.gradient_g1(self.u_old).scalar_mult (u - self.u_old)

    def run_count(self):
        #find u1
        u_k = self.grad_method(self.u_old.mul_num(0.9))

        print('W_k 0', self.w_k(self.u_old))
        print('first run ok')
        print(abs(self.f(u_k) - self.f(self.u_old)))
        while(abs(self.f(u_k) - self.f(self.u_old)) > self.eps):
            temp = u_k
            u_k = self.grad_method(u_k)
            self.u_old = temp
            print('main loop end')
        print('Решение задачи: ', u_k)
        return u_k

    def grad_method(self, u_n):
        n = 1
        alpha = 1/n
        grad = self.gradient_phi(u_n, self.u_old)
        while grad.normalize() > self.eps:
            u_n_new = u_n - grad * alpha
            n += 1
            alpha = 1/n
            grad = self.gradient_phi(u_n_new, u_n)
            print('grad --', grad, ' alpha --', alpha, ' u_n --', u_n)
            print(grad.normalize())
            print('grad loop end')
            u_n = u_n_new
        return u_n

    def gradient_phi(self, u, u_old):
        grad = self.gradient_f(u_old).mul_num(self.BETTA) +(u-u_old)
        return grad

    def f(self, vect):
        # целевая функция x1^2+ x2^2 --> min
        return vect.x**2 + vect.y**2

    def g1(self, vect):
        # заданное ограничение x1^2 - x2 >= 0
        return vect.x ** 2 - vect.y


    def gradient_f(self, vect):
        return Vector2D(2*vect.x, 2*vect.y)

    def gradient_g1(self, vect):
        return Vector2D(2 * vect.x, - 1)


    def run_method(self):
        print('For more info about program press 1')
        while 1:
            a = input('>>')
            if a == '1':
                print('You can do the following instructions:\n1 - press for help\n2 - press to change start point (enter first coordinate, then press enter)\n3 - to change epsilon(please enter with dot)\n4 - to change betta\n5 - to start counting1\n6 - to print function\n7 - to exit')
            elif a == '2':
                x = int(input('>>'))
                y = int(input('>>'))
                self.u_old = Vector2D(x,y)
            elif a == '3':
                self.eps = float(input('>>'))
            elif a == '4':
                self.BETTA = float1(input('>>'))
            elif a == '5':
                self.run_count()
            elif a == '6':
                print('J(u) = x1^2+ x2^2 --> min')
            elif a == '7':
                exit()
            else:
                print('Wrong command, please try again')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    li = linearization()
    li.run_method()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/