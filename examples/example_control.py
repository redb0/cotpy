import numpy as np

import matplotlib.font_manager
import matplotlib.pyplot as plt

from cotpy import model
from cotpy.identification import alg
from cotpy.identification import identifier
from cotpy.control.regulator import Regulator


del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
params = {'font.family': 'Times New Roman',
          'font.weight': 'normal',
          'font.size': 14,
          'mathtext.fontset': 'custom',
          'mathtext.rm': 'Times New Roman',
          'mathtext.it': 'Times New Roman:italic',
          'text.usetex': False,  # True for use latex
          'mathtext.bf': 'Times New Roman:bold',
          }
plt.rcParams.update(params)


def draw(input_x, output, a, model_v, real_a, setpoint, time):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(time, output, label='$x(t)$')
    axs[0].plot(time, setpoint, label='$x^{*}(t)$', linestyle='--')
    # axs[0].plot(time, model_v, label='Модель')  # Model
    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('$x(t),x^{*}(t)$', rotation='horizontal', ha='right')
    axs[0].grid(True)
    l = axs[0].legend(labelspacing=0, borderpad=0.1)
    l.draggable(True)

    axs[1].plot(time, input_x, label='Управление')
    axs[1].set_xlabel('$t$')
    axs[1].set_ylabel('$u(t)$', rotation='horizontal', ha='right')
    axs[1].grid(True)
    plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # if a is not None:
    #     for i in range(1, len(a[0])):
    #         axs[i].plot(time, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
    #         axs[i].plot(time, [real_a[i] for _ in range(len(a))], linestyle='--',
    #                     color=colors[i])  # label=f'real a_{i}',
    #         axs[i].set_xlabel('Время')  # Time
    #         axs[i].grid(True)
    #         l = axs[i].legend()
    #         l.draggable(True)
    #
    #     axs[1].set_xlabel('Время')  # Time
    #     axs[1].grid(True)
    #     l = axs[1].legend()
    #     l.draggable(True)

    axs[0].plot(time, a[:, 0], linestyle='-', color=colors[0])  # label=f'a{0}'
    axs[0].plot(time, [real_a[0] for _ in range(len(a))], linestyle='--', color=colors[0])  # label=f'real a_{i}',
    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('$\\alpha_{0}$', rotation='horizontal', ha='right')
    axs[0].grid(True)

    axs[1].plot(time, a[:, 1], label='$\\alpha_{1}$', linestyle='-', color=colors[1])
    axs[1].plot(time, [real_a[1] for _ in range(len(a))], linestyle='--', color=colors[1])
    axs[1].plot(time, a[:, 2], label='$\\alpha_{2}$', linestyle='-', color=colors[2])
    axs[1].plot(time, [real_a[2] for _ in range(len(a))], linestyle='--', color=colors[2])
    axs[1].set_xlabel('$t$')
    axs[1].set_ylabel('$\\alpha_{1},\\alpha_{2}$', rotation='horizontal', ha='right')
    axs[1].grid(True)
    l = axs[1].legend(labelspacing=0, borderpad=0.3)
    l.draggable(True)
    axs[2].plot(time, a[:, 3], label='$\\alpha_{3}$', linestyle='-', color=colors[3])
    axs[2].plot(time, [real_a[3] for _ in range(len(a))], linestyle='--', color=colors[3])
    axs[2].plot(time, a[:, 4], label='$\\alpha_{4}$', linestyle='-', color=colors[4])
    axs[2].plot(time, [real_a[4] for _ in range(len(a))], linestyle='--', color=colors[4])
    axs[2].set_xlabel('$t$')
    axs[2].set_ylabel('$\\alpha_{3},\\alpha_{4}$', rotation='horizontal', ha='right')
    axs[2].grid(True)
    l = axs[2].legend(labelspacing=0, borderpad=0.3)
    l.draggable(True)
    plt.show()


def example_1():
    """Пример использования библиотеки CotPy 
    для синтеза закона адаптивного управления 
    с идентификацией. Используется модель с 
    чистым запаздыванием в 2 такта."""

    def obj(x, u):
        """Функция имитации объекта."""
        return -10 + 0.3 * x + 3 * u + np.random.uniform(-2, 2)

    def xt(j):
        """Функция генерации уставки."""
        if j < 10:
            return 100
        elif 10 <= j <= 15:
            return 150
        else:
            return 50

    # создание модели (с чистым запаздыванием)
    m = model.create_model('a_0+a1*x1(t-1)+a_2*u1(t-3)')

    # создание идентификатора и инициализация начальных значений
    idn = identifier.Identifier(m)
    idn.init_data(x=[[0, -11.33, -9.37]],
                  a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  u=[[0, 2, 3, 4, 4]])

    # определение алгоритма идентификации
    lsm = alg.Adaptive(idn, m='lsm')
    # smp = alg.Adaptive(idn, m='smp')

    # массивы для сохранения промежуточных значений
    u_ar = np.zeros((25,))
    o_ar = np.zeros((25,))
    m_ar = np.zeros((25,))
    a_ar = np.zeros((25, 3))
    xt_ar = np.zeros((25, ))
    er = np.zeros((25, ))

    # создание регулятора, установка ограничений на управление и синтез закона управления
    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()

    # основной цикл
    for i in range(25):
        # измерение выхода объекта
        v = obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])
        obj_val = [v]

        # идентификация коэффициентов модели и обновляем значения коэффициентов
        new_a = lsm.update(obj_val, w=0.01, init_weight=0.01)
        # new_a = smp.update(obj_val, gamma=0.01, gt='a', weight=0.9, h=0.1, deep_tuning=True)
        idn.update_a(new_a)

        # расчет управляющего воздействия
        new_u = r.update(obj_val, xt(i+3))

        # сохранение промежуточных результатов для построения графика
        u_ar[i] = new_u[0]
        o_ar[i] = obj_val[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)
        m_ar[i] = idn.model.get_last_model_value()

        # обновление состояния модели
        idn.update_x(obj_val)
        idn.update_u(new_u)
        er[i] = v - xt(i)

    print(m.last_a)
    print(er)
    t = np.array([i for i in range(25)])
    draw(u_ar, o_ar, a_ar, m_ar, [-10, 0.3, 3], xt_ar, t)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(t, er, label='Error')
    axs.set_xlabel('Time')
    axs.grid(True)
    axs.legend()
    plt.show()


def example_2():
    """Пример использования библиотеки CotPy для синтеза 
    закона адаптивного управления. Для идентификации 
    используется алгоритм метода наименьших квадратов. 
    Модель простая с небольшой инерцией."""

    def obj(x1, x2, u):
        """Функция имитации объекта."""
        return -10 + 0.3 * x1 + 0.1 * x2 + 3 * u + np.random.uniform(-5, 5)

    def xt(j):
        """Функция генерации уставки."""
        if j < 10:
            return 10
        elif 10 <= j < 15:
            return 100
        elif 15 <= j <= 20:
            return 250
        else:
            return 50

    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-1)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)

    # значения коэффициентов 'a' выставляем равными 1 (можно не
    # передавать вовсе, тогда по умолчанию инициализируется единицами)
    # либо другими предполагаемыми значениями.
    idn.init_data(x=[[0, 0, -10, -10, -11]], u=[[0, 1, 1, 1]])
    lsm = alg.Adaptive(idn, m='lsm')

    r = Regulator(m)
    r.set_limit(0, 255)
    r.synthesis()

    u_ar = np.zeros((30,))
    o_ar = np.zeros((30,))
    m_ar = np.zeros((30,))
    a_ar = np.zeros((30, 4))
    xt_ar = np.zeros((30,))

    for i in range(30):
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]

        new_a = lsm.update(x, w=0.01, init_weight=0.01)
        idn.update_a(new_a)

        new_u = r.update(x, xt(i + 1))

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x)
        idn.update_u(new_u)

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(30)])
    draw(u_ar, o_ar, a_ar, m_ar, [-10, 0.3, 0.1, 3], xt_ar, t)


def example_3():
    """Пример использование библиотеки CotPy для адаптивного 
    управления на основе модели без идентификации.
    Модель с инерцией по управлению."""

    def obj(x, u1, u2):
        """Функция имитации объекта."""
        return x + u1 + u2

    def xt(j):
        """Функция генерации уставки."""
        if j < 10:
            return 10
        elif 10 <= j < 15:
            return 100
        elif 15 <= j <= 20:
            return 250
        else:
            return 50

    m = model.create_model("x(t-1)+u(t-1)+u(t-2)")
    m.initialization(x=[[0]], u=[[0, 0]])

    r = Regulator(m)
    r.synthesis()
    r.set_limit(-25, 40)

    u_ar = np.zeros((30,))
    o_ar = np.zeros((30,))
    xt_ar = np.zeros((30,))
    m_ar = np.zeros((30,))

    for i in range(30):
        obj_val = [obj(*m.get_var_values(t='x')[0], *m.get_var_values(t='u')[0])]
        new_u = r.update(obj_val, xt(i + 1))
        m.update_x(obj_val)
        m.update_u(new_u)
        u_ar[i] = new_u[0]
        o_ar[i] = obj_val[0]
        xt_ar[i] = xt(i)
        m_ar[i] = m.get_last_model_value()

    t = np.array([i for i in range(30)])
    draw(u_ar, o_ar, None, m_ar, [], xt_ar, t)


def example_4(use_lsm=False):
    """Пример использование библиотеки CotPy для адаптивного 
    управления на основе сложной линейной модели с чистым 
    запаздыванием в 5 тактов.
    Моделируется процесс нагрева и поддержания температуры жидкости."""

    def obj(x1, x2, u1, u2):
        return 1.68 + 1.235 * x1 - 0.319 * x2 + 0.04 * u1 + 0.027 * u2  # + np.random.uniform(-0.1, 0.1)

    def xt(j):
        if j <= 50:
            return 80
        elif 50 < j <= 80:
            return 35
        elif 80 < j <= 100:
            return 25
        elif 100 < j <= 125:
            return 60
        elif 125 < j <= 150:
            return 70
        elif 150 < j <= 175:
            return 20
        elif 175 < j <= 200:
            return 90
        else:
            return 65

    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-6)+a4*u(t-7)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)
    print('Модель:', m.sp_expr)

    if use_lsm:
        init_x = [[20, 20, 20.4, 20.764, 21.246, 21.528]]
        init_u = [[10., 0., 10., 0., 10., 0., 0., 0., 0., 0., 0.]]
        method = 'lsm'
    else:
        init_x = np.full((1, 6), 20.)
        init_u = np.zeros((1, 11))
        method = 'smp'

    # TODO: сделать для случая когда память меньше memory_size = 1
    idn.init_data(a=np.ones((5, 5)), x=init_x, u=init_u, type_memory='min', memory_size=5)
    algorithm = alg.Adaptive(idn, m=method)

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()
    print('Закон управления:', r.expr)
    print('Аргументы:', r.expr_args)

    n = 240
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 5))
    xt_ar = np.zeros((n,))

    a = np.zeros((n, 5))

    for i in range(n):
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]
        # Используется глубокая подстройка на основе приращений (deep_tuning=True),
        # а также адаптивный вес при подстройке свободного коэффициента (aw=True)
        if use_lsm:
            new_a = algorithm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=False)
        else:
            new_a = algorithm.update(x, gamma=1, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True, use_memory=True)

        idn.update_a(new_a)

        new_u = r.update(x, xt(i + 6))

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x)
        idn.update_u(new_u)

        a[i] = np.array([1.68, 1.235, -0.319, 0.04, 0.027]) - new_a

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, [1.68, 1.235, -0.319, 0.04, 0.027], xt_ar, t)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if a is not None:
        for i in range(len(a[0])):
            axs.plot(t, a[:, i], label=f'$a_{i}$', linestyle='-', color=colors[i])
        axs.set_xlabel('$t$')  # Time
        axs.grid(True)
        axs.legend()

    plt.show()


def example_5():
    """Пример использование библиотеки CotPy для адаптивного 
    управления на основе нелинейной модели с чистым 
    запаздыванием в 1 такт."""
    def obj(x, u):
        return 1.3 + 0.52 * x * u  # + np.random.uniform(-0.1, 0.1)

    def xt(j):
        if j <= 75:
            return 75
        elif 75 < j <= 100:
            return 35
        elif 100 < j <= 120:
            return 25
        else:
            return 60

    expr = "a0+a1*x(t-1)*u(t-2)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)

    idn.init_data(a=[[1, 1], [1, 1]],
                  x=[[1.3, 1.3]],
                  u=[[0, 0, 0]])
    smp = alg.Adaptive(idn, m='smp')

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()

    n = 240
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 2))
    xt_ar = np.zeros((n,))

    for i in range(n):
        x_val = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]

        new_a = smp.update(x_val, gamma=1, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True)
        idn.update_a(new_a)

        new_u = r.update(x_val, xt(i + 1))

        u_ar[i] = new_u[0]
        o_ar[i] = x_val[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x_val)
        idn.update_u(new_u)

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, [1.3, 0.52], xt_ar, t)


def example_6():
    def obj(x1, x2, u1, u2):
        return 1.68 + 1.235 * x1 - 0.319 * x2 + 0.04 * u1 + 0.027 * u2

    t = np.array([i for i in range(50)])
    x = np.zeros((50,))
    for i in range(50):
        if i < 2:
            x[i] = obj(20, 20, 100, 100)
        else:
            x[i] = obj(x[i-1], x[i-2], 100, 100)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(t, x, label='x', linestyle='-', )
    axs.set_xlabel('Время')  # Time
    axs.grid(True)
    axs.legend()

    plt.show()


def example_7():
    def obj(x1, u1):
        # return 4 + 1.2 * x1 + 0.7 * u1
        return -4 + 0.8*x1 + 0.5*u1 #+ np.random.uniform(-0.1, 0.1)

    expr = "a0+a1*x(t-1)+a2*u(t-1)"  # "a0+a1*x(t-1)+a2*u(t-1)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)

    # idn.init_data(a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #               x=[[0, 4, 9.5, 15.4, 22.83]],
    #               u=[[0, 1, 0, 0.5, 0]], memory_size=5, type_memory='max')
    idn.init_data(a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  x=[[0, 3.93, 7.55, 10.01, 12.35]],
                  u=[[0, 1, 0, 0.5, 0]], memory_size=3, type_memory='max')
    smp = alg.Adaptive(idn, m='smp')
    lsm = alg.Adaptive(idn, m='lsm')

    r = Regulator(m)
    r.set_limit(-50, 50)
    r.synthesis()

    print('() ->', m.get_var_values(t='output'))
    print('0 ->', m.get_var_values(t='output', n=0))
    print('-1 ->', m.get_var_values(t='output', n=-1))
    print('-2 ->', m.get_var_values(t='output', n=-2))
    # print('-3 ->', m.get_x_values(n=-3))
    print('--------')
    print('1 ->', m.get_var_values(t='output', n=1))
    print('2 ->', m.get_var_values(t='output', n=2))
    print('3 ->', m.get_var_values(t='output', n=3))
    # print('4 ->', m.get_x_values(n=4))
    # print('3 ->', m.get_x_values(n=3))

    n = 100
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 3))
    xt_ar = np.zeros((n,))
    a = np.zeros((n, 3))

    for i in range(n):
        print(i)
        # x = [obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])]
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]
        # new_a = smp.update(x, gamma=0.9, gt='f', weight=0.9, h=0.1, deep_tuning=True, aw=False, use_memory=False)
        new_a = lsm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=False)

        idn.update_a(new_a)
        new_u = r.update(x, 100)

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = 100
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x)
        idn.update_u(new_u)
        print('new_a:', new_a)
        print('----------------------------------------------------------------')

        a[i] = np.array([-4, .8, 0.5]) - new_a

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, [-4, 0.8, 0.5], xt_ar, t)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if a is not None:
        for i in range(len(a[0])):
            axs.plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        axs.set_xlabel('Время')  # Time
        axs.grid(True)
        axs.legend()

    plt.show()


def example_my_data(show_init_data=False):

    a_slsm = np.array([-0.19973176, 1.748009, -0.74896242, 0.00832433, -0.0038998])  # По статическому методу МНК
    a_slsm1 = np.array([0.03667119, 1.77382635, -0.77572604, 0.00631385, -0.00400434])  # По статическому методу МНК (с удержанием температуры на 27.4)
    a1 = np.array([0.0263, 1.748, -0.74896, 0.008324, - 0.0038998])  # Удерживает начальную температуру, больше ошибка приближения

    def obj(x1, x2, u1, u2, a):
        return a[0] + a[1] * x1 + a[2] * x2 + a[3] * u1 + a[4] * u2

    if show_init_data:
        p = []
        t1 = []
        with open('data1.txt', 'r') as f:
            _t = 27.4
            for line in f.readlines():
                if line[0] == 'P':
                    _p = int(line.split(' ')[-1])
                    p.append(_p)
                    t1.append(_t)
                    if t1[-1] == -127:
                        t1[-1] = t1[-2]
                if line[0] == 'T':
                    _t = int(line.split(' ')[-2]) / 10.0
        print(len(p))
        print(len(t1))

        # -------- График исходных данных, модели и ошибки -----------
        n = len(t1)
        t = np.array([i for i in range(n)])
        x_data = np.zeros((n, 2))
        e_data = np.zeros((n, 2))
        for i in range(n):
            if i <= 1:
                x_data[i][0] = obj(t1[0], t1[0], 0, 0, a_slsm)
                x_data[i][1] = obj(t1[0], t1[0], 0, 0, a_slsm1)
            else:
                x_data[i][0] = obj(t1[i-1], t1[i - 2], p[i-1], p[i - 2], a_slsm)
                x_data[i][1] = obj(t1[i-1], t1[i - 2], p[i-1], p[i - 2], a_slsm1)
            e_data[i][0] = t1[i] - x_data[i][0]
            e_data[i][1] = t1[i] - x_data[i][1]

        fig, axs = plt.subplots(nrows=4, ncols=1)
        axs[0].plot(t, x_data[:, 0], label='Модель по стат. МНК')
        axs[0].plot(t, t1, label='Объект')
        axs[1].plot(t, e_data[:, 0], label='Ошибка 1')
        axs[2].plot(t, x_data[:, 1], label='Модель 2')
        axs[2].plot(t, t1, label='Объект')
        axs[3].plot(t, e_data[:, 1], label='Ошибка 2')

        for i in range(4):
            axs[i].set_xlabel('Время')
            axs[i].grid(True)
            axs[i].legend()
        plt.show()
        # ------------------------------------------------------------

    def get_xt(j, is_imitation=False):
        if is_imitation:
            if j <= 100:  # просто имитация
                return 75
            elif 100 < j <= 200:
                return 78.4
            elif 200 < j <= 300:
                return 97.7
            elif 300 < j <= 350:
                return 60
            elif 350 < j <= 400:
                return 70
            elif 400 < j <= 500:
                return 40
            elif 500 < j <= 600:
                return 90
            else:
                return 55
        else:
            if j <= 70:  # уставка для рабочего процесса
                return 81.4
            else:
                return 0.084*j+75.52

    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-1)+a4*u(t-2)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)
    print('Модель:', m.sp_expr)

    idn.init_data(a=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                  #x=[[27.4, 27.4, 28.1, 30.0, 30.8, 31.1]],
                  #u=[[0, 100, 100, 100, 100, 100]], type_memory='min', memory_size=5
                  x=[[27.4, 27.4, 27.4, 27.4, 27.4, 27.4]],
                  u=[[0, 0, 0, 0, 0, 0]], type_memory='min', memory_size=5
                  )
    smp = alg.Adaptive(idn, m='smp')
    lsm = alg.Adaptive(idn, m='lsm')

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()
    print('Закон управления:', r.expr)
    print('Аргументы:', r.expr_args)

    n = 270  # 700
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, len(a_slsm)))
    xt_ar = np.zeros((n,))

    a_er = np.zeros((n, len(a_slsm)))

    for i in range(n):
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0], a_slsm1)]

        new_a = smp.update(x, gamma=0.01, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True, use_memory=True)
        # new_a = lsm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=True)  # efi=0.9

        idn.update_a(new_a)

        new_u = r.update(x, get_xt(i + 1, False))

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = get_xt(i, False)
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x)
        idn.update_u(new_u)

        a_er[i] = a_slsm1 - new_a

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, a_slsm1, xt_ar, t)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if a_er is not None:
        for i in range(len(a_er[0])):
            axs.plot(t, a_er[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        axs.set_xlabel('Время')  # Time
        axs.grid(True)
        axs.legend()

    plt.show()


def example_9(use_lsm=False):

    def obj(x1, x2, u1, u2, z1, z2):
        return 1.68 + 1.235 * x1 - 0.319 * x2 + 0.04 * u1 + 0.027 * u2 + 0.1*z1+0.05*z2

    def xt():
        return 65

    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-6)+a4*u(t-7)+a5*z1(t-1)+a6*z2(t-1)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)
    print('Модель:', m.sp_expr)

    # TODO: сделать для случая когда память меньше memory_size = 1
    if use_lsm:
        init_x = [[20, 20, 20.9885, 22.27, 23.39, 24.57, 25.51, 26.06]]
        init_u = [[10, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]]
        init_z = np.array([[5.765, 5.24, 5.00, 5.77, 5.32, 5.90, 5.13], [2.84, 2.54, 2.76, 2.51, 2.95, 2.52, 2.59]])
        method = 'lsm'
    else:
        init_x = np.full((1, 8), 20.)
        init_u = np.zeros((1, 13))
        init_z = np.array([np.random.uniform(5, 6, 7), np.random.uniform(2, 2.5, 7)])
        method = 'smp'

    idn.init_data(a=np.ones((7, 7)), x=init_x, u=init_u, z=init_z,
                  type_memory='min', memory_size=5)
    algorithm = alg.Adaptive(idn, m=method)

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()
    print('Закон управления:', r.expr)
    print('Аргументы:', r.expr_args)

    n = 240
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 7))
    xt_ar = np.zeros((n,))

    a = np.zeros((n, 7))

    for i in range(n):
        x = [obj(*np.hstack(idn.model.get_var_values(t='x')),
                 *np.hstack(idn.model.get_var_values(t='u')), *np.hstack(idn.model.get_var_values(t='z')))]
        # Используется глубокая подстройка на основе приращений (deep_tuning=True),
        # а также адаптивный вес при подстройке свободного коэффициента (aw=True)
        if use_lsm:
            new_a = algorithm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=False)
        else:
            new_a = algorithm.update(x, gamma=1, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True, use_memory=True)

        idn.update_a(new_a)
        z = [np.random.uniform(5, 6), np.random.uniform(2.5, 3)]
        new_u = r.update(x, xt(), ainputs=z)

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt()
        m_ar[i] = idn.model.get_last_model_value()

        idn.update_x(x)
        idn.update_u(new_u)
        idn.update_z(z)

        a[i] = np.array([1.68, 1.235, -0.319, 0.04, 0.027, 0.1, 0.05]) - new_a

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, [1.68, 1.235, -0.319, 0.04, 0.027, 0.1, 0.05], xt_ar, t)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if a is not None:
        for i in range(len(a[0])):
            axs.plot(t, a[:, i], label=f'$a_{i}$', linestyle='-', color=colors[i])
        axs.set_xlabel('$t$')  # Time
        axs.grid(True)
        axs.legend()

    plt.show()


def main():
    # Раскомментируйте интересующий пример.
    # example_1()
    # example_2()
    # example_3()
    example_4()
    # example_5()
    # example_6()

    # example_7()
    # example_8()
    # lsm()
    # example_my_data(False)
    # example_9()

if __name__ == '__main__':
    main()
