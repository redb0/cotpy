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


def plot_coefficient(axs, t, a, ax, xlabel, ylabel, color):
    axs.plot(t, a, linestyle='-', color=color)
    axs.plot(t, [ax for _ in range(len(a))], linestyle='--', color=color)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel, rotation='horizontal', ha='right')
    axs.grid(True)


def draw_object_movement(t, x, xt, u):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(t, x, label='$x(t)$')
    axs[0].plot(t, xt, label='$x^{*}(t)$', linestyle='--')
    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('$x(t),x^{*}(t)$', rotation='horizontal', ha='right')
    axs[0].grid(True)
    l = axs[0].legend(labelspacing=0, borderpad=0.1)
    l.set_draggable(True)
    axs[1].plot(t, u)
    axs[1].set_xlabel('$t$')
    axs[1].set_ylabel('$u(t)$', rotation='horizontal', ha='right')
    axs[1].grid(True)
    plt.show()


def example_1(with_err=False):
    """
    Пример использования библиотеки CotPy 
    для синтеза закона адаптивного управления 
    с идентификацией. Используется модель с 
    чистым запаздыванием в 2 такта.
    :param with_err: если True добавляется аддитивная равномерно распределенная помеха 
                     амплитудой 0.2 к измерению выхода объекта
    :return: None
    """

    a = [-10, 0.3, 3]  # реальные коэффициенты

    def obj(x, u):
        """Функция имитации объекта."""
        val = a[0] + a[1] * x + a[2] * u
        if with_err:
            return val + np.random.uniform(-0.1, 0.1)
        else:
            return val

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
                  a=np.ones((3, 3)),
                  u=[[0, 2, 3, 4, 4]])

    # определение алгоритма идентификации
    smp = alg.Adaptive(idn, m='smp')

    n = 30
    # массивы для сохранения промежуточных значений
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    a_ar = np.zeros((n, len(a)))
    xt_ar = np.zeros((n, ))

    # создание регулятора, установка ограничений на управление и синтез закона управления
    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()

    # основной цикл
    for i in range(n):
        # измерение выхода объекта
        obj_val = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]

        # Для идентификации используется простейший адаптивный алгоритм
        # с методом последовательной линеаризации (глубокая подстройка)
        new_a = smp.update(obj_val, gamma=0.01, gt='a', weight=0.9, h=0.1, deep_tuning=True)
        idn.update_a(new_a)

        # расчет управляющего воздействия
        new_u = r.update(obj_val, xt(i+3))

        # сохранение промежуточных результатов для построения графика
        u_ar[i] = new_u[0]
        o_ar[i] = obj_val[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)

        # обновление состояния модели
        idn.update_x(obj_val)
        idn.update_u(new_u)

    t = [i for i in range(n)]
    draw_object_movement(t, o_ar, xt_ar, u_ar)

    # График процесса идентификации
    fig, axs = plt.subplots(nrows=3, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(len(a)):
        plot_coefficient(axs[i], t, a_ar[:, i], a[i], '$t$', f'$\\alpha_{{{str(i)}}}$', colors[i])

    plt.show()


def example_2(with_err=False):
    """
    Пример использования библиотеки CotPy для синтеза 
    закона адаптивного управления. 
    Для идентификации используется стандартный алгоритм метода наименьших квадратов. 
    Модель простая с небольшой инерцией.
    :param with_err: если True добавляется аддитивная равномерно распределенная помеха 
                     амплитудой 0.4 к измерению выхода объекта
    :return: None
    """

    a = [-10, 0.3, 0.1, 3]

    def obj(x1, x2, u):
        """Функция имитации объекта."""
        val = a[0] + a[1] * x1 + a[2] * x2 + a[3] * u
        if with_err:
            return val + np.random.uniform(-0.2, 0.2)
        else:
            return val

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

    n = 30
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    a_ar = np.zeros((n, len(a)))
    xt_ar = np.zeros((n,))

    for i in range(n):
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]

        new_a = lsm.update(x, w=0.01, init_weight=0.01)
        idn.update_a(new_a)

        new_u = r.update(x, xt(i + 1))

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)

        idn.update_x(x)
        idn.update_u(new_u)

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])

    draw_object_movement(t, o_ar, xt_ar, u_ar)

    # График процесса идентификации
    fig, axs = plt.subplots(nrows=3, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plot_coefficient(axs[0], t, a_ar[:, 0], a[0], '$t$', '$\\alpha_{0}$', colors[0])

    for i in range(1, 3):
        axs[1].plot(t, a_ar[:, i], label=f'$\\alpha_{{{str(i)}}}$', linestyle='-', color=colors[i])
        axs[1].plot(t, [a[i] for _ in range(n)], linestyle='--', color=colors[i])
    axs[1].set_xlabel('$t$')
    axs[1].set_ylabel('$\\alpha_{1},\\alpha_{2}$', rotation='horizontal', ha='right')
    axs[1].grid(True)
    l = axs[1].legend(labelspacing=0, borderpad=0.3)
    l.set_draggable(True)

    plot_coefficient(axs[2], t, a_ar[:, 3], a[3], '$t$', '$\\alpha_{3}$', colors[1])

    plt.show()


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

    n = 30
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    xt_ar = np.zeros((n,))

    for i in range(n):
        obj_val = [obj(*m.get_var_values(t='x')[0], *m.get_var_values(t='u')[0])]
        new_u = r.update(obj_val, xt(i + 1))
        m.update_x(obj_val)
        m.update_u(new_u)
        u_ar[i] = new_u[0]
        o_ar[i] = obj_val[0]
        xt_ar[i] = xt(i)

    t = np.array([i for i in range(n)])

    draw_object_movement(t, o_ar, xt_ar, u_ar)


def example_4(use_lsm=False, with_err=False):
    """
    Пример использование библиотеки CotPy для адаптивного 
    управления на основе сложной линейной модели с чистым 
    запаздыванием в 5 тактов.
    Моделируется процесс нагрева и поддержания температуры жидкости.
    :param use_lsm: Использование адаптивного алгоритма МНК. 
                    Есле use_lsm=False используется простейшый адаптивный алгоритм.
    :param with_err: Добавление аддитивной равномерно распределенной помехи.
    :return: None
    """

    # реальные коэффициенты
    a = [1.68, 1.235, -0.319, 0.04, 0.027]

    def obj(x1, x2, u1, u2):
        """Имитация объекта"""
        val = a[0] + a[1] * x1 + a[2] * x2 + a[3] * u1 + a[4] * u2
        if with_err:
            return val + np.random.uniform(-0.1, 0.1)
        else:
            return val

    def xt(j):
        """Уставка"""
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

    # Создание модели и идентификатора
    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-6)+a4*u(t-7)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)
    print('Модель:', m.sp_expr)
    if m.get_index_fm() is not None:
        print(f'Индекс свободного члена: {m.get_index_fm()}')
    else:
        print('Нет свободного члена')

    # Инициализация начальных значений
    if use_lsm:
        # Для адаптивного алгоритма МНК необходимы "исторические" данные работы объекта.
        # Инициализация управления нулями (при выключенном объекте)
        # приведет к неработоспособности алгоритма.
        init_x = [[20, 20, 20.4, 20.764, 21.246, 21.528]]
        init_u = [[10., 0., 10., 0., 10., 0., 0., 0., 0., 0., 0.]]
        method = 'lsm'
    else:
        # Простейший адаптивный алгоритм не чувствителен к начальным данным.
        init_x = np.full((1, 6), 20.)
        init_u = np.zeros((1, 11))
        method = 'smp'

    idn.init_data(a=np.ones((5, 5)), x=init_x, u=init_u, type_memory='min', memory_size=5)
    algorithm = alg.Adaptive(idn, m=method)

    # Создание регулятора и синтез закона управления
    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()
    print('Закон управления:', r.expr)
    print('Аргументы:', r.expr_args)

    n = 240  # количество рабочих тактов
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    a_ar = np.zeros((n, len(a)))
    xt_ar = np.zeros((n,))
    a_er = np.zeros((n, len(a)))  # массив для сохранения ошибки идентификации

    for i in range(n):
        x = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]
        # Идентификация коэффициентов
        # Используется глубокая подстройка на основе приращений (deep_tuning=True),
        # а также адаптивный вес при подстройке свободного коэффициента (aw=True)
        if use_lsm:
            # Использование модели в приращениях uinc=True и
            # адаптивного веса при подстройке свободного коэффициента adaptive_weight=False
            new_a = algorithm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=False)
        else:
            # Использование глубокой подстройки (с использованием модели в приращениях) deep_tuning=True,
            # адаптивного веса aw=True,
            # использование памяти use_memory=True
            new_a = algorithm.update(x, gamma=1, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True, use_memory=True)

        idn.update_a(new_a)

        # Расчет управляющего воздействия ведется с учетом запаздывания в 5 тактов.
        new_u = r.update(x, xt(i + 6))

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)
        a_er[i] = np.array(a) - new_a

        idn.update_x(x)
        idn.update_u(new_u)

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])

    # График движенияя объекта
    draw_object_movement(t, o_ar, xt_ar, u_ar)

    # График процесса идентификации
    fig, axs = plt.subplots(nrows=3, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plot_coefficient(axs[0], t, a_ar[:, 0], a[0], '$t$', '$\\alpha_{0}$', colors[0])

    n_ax = 1
    for i in range(1, 5, 2):
        for k in range(2):
            axs[n_ax].plot(t, a_ar[:, i+k], label=f'$\\alpha_{{{str(i+k)}}}$', linestyle='-', color=colors[k+1])
            axs[n_ax].plot(t, [a[i+k] for _ in range(n)], linestyle='--', color=colors[k+1])
        axs[n_ax].set_xlabel('$t$')
        axs[n_ax].set_ylabel(f'$\\alpha_{{{str(i)}}},\\alpha_{{{str(i + 1)}}}$', rotation='horizontal', ha='right')
        axs[n_ax].grid(True)
        l = axs[n_ax].legend(labelspacing=0, borderpad=0.3)
        l.set_draggable(True)
        n_ax += 1

    # График ошибки идентификации
    fig, axs = plt.subplots(nrows=1, ncols=1)
    if a_er is not None:
        for i in range(len(a_er[0])):
            axs.plot(t, a_er[:, i], label=f'$e_{{\\alpha_{i}}}$', linestyle='-', color=colors[i])
        axs.set_xlabel('$t$')
        axs.grid(True)
        axs.legend()

    plt.show()


def example_5(with_err=False):
    """
    Пример использование библиотеки CotPy для адаптивного 
    управления на основе нелинейной модели с чистым 
    запаздыванием в 1 такт.
    :param with_err: Добавление аддитивной равномерно 
                     распределенной помехи амплитудой 0.2 к выходу объекта 
    :return: None
    """

    a = [1.3, 0.52]

    def obj(x, u):
        """Имитация объекта, описываемого нелинейной относительно входов/выходов моделью."""
        val = a[0] + a[1] * x * u
        if with_err:
            return val + np.random.uniform(-0.1, 0.1)
        return val

    def xt(j):
        """Уставка"""
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

    idn.init_data(a=np.ones((2, 2)),
                  x=[[1.3, 1.3]],
                  u=[[0, 0, 0]])
    # В данном случае простейший адаптивный алгоритм отлично справиться
    smp = alg.Adaptive(idn, m='smp')

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()

    n = 240
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    a_ar = np.zeros((n, len(a)))
    xt_ar = np.zeros((n,))

    for i in range(n):
        x_val = [obj(*idn.model.get_var_values(t='x')[0], *idn.model.get_var_values(t='u')[0])]

        new_a = smp.update(x_val, gamma=1, gt='a', weight=0.9, h=0.1, deep_tuning=False, aw=False)
        idn.update_a(new_a)

        new_u = r.update(x_val, xt(i + 1))

        u_ar[i] = new_u[0]
        o_ar[i] = x_val[0]
        a_ar[i] = new_a
        xt_ar[i] = xt(i)

        idn.update_x(x_val)
        idn.update_u(new_u)

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])

    # График движенияя объекта
    draw_object_movement(t, o_ar, xt_ar, u_ar)

    # График процесса идентификации
    fig, axs = plt.subplots(nrows=2, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plot_coefficient(axs[0], t, a_ar[:, 0], a[0], '$t$', '$\\alpha_{0}$', colors[0])
    plot_coefficient(axs[1], t, a_ar[:, 1], a[1], '$t$', '$\\alpha_{1}$', colors[0])


def example_6(use_lsm=False, with_err=False):
    """
    Пример использования библиотеки CotPy для модели с дополнительными входами.
    :param use_lsm: Если уставновлен True используется адаптивный МНК, 
                    иначе простейший адаптивный алгоритм.
    :param with_err: Добавление аддитивной равномерно 
                     распределенной помехи амплитудой 0.1 к выходу объекта 
    :return: None
    """

    a = np.array([1.68, 1.235, -0.319, 0.04, 0.027, 0.1, 0.05])

    def obj(x1, x2, u1, u2, z1, z2):
        """Имитация объекта"""
        val = a[0] + a[1] * x1 + a[2] * x2 + a[3] * u1 + a[4] * u2 + a[5] * z1 + a[6] * z2
        if with_err:
            return val + np.random.uniform(-0.05, 0.05)
        else:
            return val

    def xt():
        """Уставка"""
        return 65

    # создание модели
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
    a_ar = np.zeros((n, 7))
    xt_ar = np.zeros((n,))

    er = np.zeros((n, 7))

    for i in range(n):
        x = [obj(*np.hstack(idn.model.get_var_values(t='x')),
                 *np.hstack(idn.model.get_var_values(t='u')), *np.hstack(idn.model.get_var_values(t='z')))]
        if use_lsm:
            # В адаптивном МНК применяется метод последовательной линеаризации (МПЛ) и адаптивный вес
            new_a = algorithm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=False)
        else:
            # Используется простейший адаптивный с памятью
            new_a = algorithm.update(x, gamma=1, gt='a', weight=0.9, h=0.1, use_memory=True)

        idn.update_a(new_a)

        # Дополнительные входы являются случайными возмущениями (которые можно только измерять)
        # и, например, принимают случайные в определенных интервалах
        z = [np.random.uniform(5, 6), np.random.uniform(2.5, 3)]
        new_u = r.update(x, xt(), ainputs=z)

        u_ar[i] = new_u[0]
        o_ar[i] = x[0]
        a_ar[i] = new_a
        xt_ar[i] = xt()

        idn.update_x(x)
        idn.update_u(new_u)
        idn.update_z(z)

        er[i] = a - new_a

    print('Last coefficients:', idn.model.last_a)

    t = np.array([i for i in range(n)])
    # График движенияя объекта
    draw_object_movement(t, o_ar, xt_ar, u_ar)

    # График процесса идентификации
    fig, axs = plt.subplots(nrows=4, ncols=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plot_coefficient(axs[0], t, a_ar[:, 0], a[0], '$t$', '$\\alpha_{0}$', colors[0])

    n_ax = 1
    for i in range(1, len(a)-1, 2):
        for k in range(2):
            axs[n_ax].plot(t, a_ar[:, i + k], label=f'$\\alpha_{{{str(i+k)}}}$', linestyle='-', color=colors[k + 1])
            axs[n_ax].plot(t, [a[i + k] for _ in range(n)], linestyle='--', color=colors[k + 1])
        axs[n_ax].set_xlabel('$t$')
        axs[n_ax].set_ylabel(f'$\\alpha_{{{str(i)}}},\\alpha_{{{str(i + 1)}}}$', rotation='horizontal', ha='right')
        axs[n_ax].grid(True)
        l = axs[n_ax].legend(labelspacing=0, borderpad=0.3)
        l.set_draggable(True)
        n_ax += 1

    plt.show()


def example_7():
    # исходные данные входов (u) и выходов (x) объекта
    u = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
         70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70]
    x = [27.4, 28.1, 30.0, 30.8, 31.1, 31.4, 31.6, 31.9, 32.1, 32.4, 32.8, 33.2, 33.7, 34.3, 34.8, 35.4, 36.1, 36.8,
         37.6, 38.3, 39.0, 39.8, 40.6, 41.4, 42.1, 42.9, 43.7, 44.4, 45.2, 46.0, 46.7, 47.5, 48.3, 49.1, 49.8, 50.6,
         51.4, 52.3, 53.1, 53.9, 54.7, 55.5, 56.3, 57.2, 58.0, 58.9, 59.8, 60.7, 61.6, 62.5, 63.4, 64.4, 65.3, 66.3,
         67.2, 68.1, 68.9, 69.9, 70.8, 71.7, 72.6, 73.6, 74.6, 75.8, 76.7, 77.7, 78.7, 79.6, 80.5, 81.4, 82.3, 83.2,
         84.0, 84.7, 85.4, 86.1, 86.6, 87.1, 87.5, 87.9, 88.3, 88.6, 88.8, 89.1, 89.4, 89.6, 89.8, 89.9, 90.1, 90.3,
         90.4, 90.5, 90.6, 90.8, 90.9, 91.0, 91.1, 91.2, 91.3, 91.4, 91.4, 91.5, 91.6, 91.6, 91.7, 91.8, 91.8, 91.9,
         91.9, 91.9, 92.1, 92.1, 92.2, 92.3, 92.3, 92.4, 92.4, 92.5, 92.5, 92.6, 92.6, 92.7, 92.7, 92.8, 92.8, 92.9,
         92.9, 92.9, 92.9, 93.0, 93.1, 93.1, 93.1, 93.1, 93.2, 93.2, 93.2, 93.2, 93.2, 93.3, 93.3, 93.3, 93.4, 93.4,
         93.4, 93.5, 93.6, 93.6, 93.6, 93.6, 93.6, 93.7, 93.7, 93.7, 93.8, 93.8, 93.8, 93.9, 93.9, 94.0, 94.0, 94.1,
         94.1, 94.2, 94.3, 94.3, 94.4, 94.4, 94.4, 94.5, 94.6, 94.6, 94.6, 94.6, 94.7, 94.8, 94.8, 94.8, 94.9, 94.9,
         94.9, 95.0, 95.0, 95.1, 95.1, 95.1, 95.2, 95.3, 95.3, 95.3, 95.4, 95.4, 95.4, 95.5, 95.5, 95.5, 95.6, 95.6,
         95.6, 95.6, 95.6, 95.6, 95.7, 95.7, 95.7, 95.7, 95.7, 95.6, 95.6, 95.6, 95.6, 95.7, 95.8, 95.8, 95.9, 95.9,
         96.0, 96.1, 96.1, 96.1, 96.2, 96.3, 96.3, 96.4, 96.4, 96.4, 96.4, 96.5, 96.5, 96.6, 96.6, 96.6, 96.7, 96.7,
         96.8, 96.8, 96.8, 96.8, 96.9, 96.9, 96.9, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.1, 97.1, 97.1, 97.2, 97.2,
         97.3, 97.3, 97.3, 97.3, 97.4, 97.4, 97.4, 97.5, 97.5, 97.5, 97.5, 97.6, 97.6, 97.6, 97.6, 97.6, 97.7, 97.7]

    # создаем модель
    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-1)+a4*u(t-2)"
    m = model.create_model(expr)
    idn = identifier.Identifier(m)
    print('Модель:', m.sp_expr)
    # подготавливаем идентификатор. В начале объект был выключен (управление равно 0) и имел температуру 27.4.
    idn.init_data(a=np.ones((5, 5)), x=np.full((1, 6), x[0]), u=np.zeros((1, 6)),
                  type_memory='min', memory_size=0)
    # для вычисления коэффициентов используем стандартный МНК
    algorithm = alg.LSM(idn, m='lsm')

    print('Количество измерений:', len(u))
    new_a = algorithm.update(u, x)
    print('Коэффициенты:', new_a)


def main():
    # Раскомментируйте интересующий пример.
    # example_1(with_err=False)
    # example_2(with_err=False)
    # example_3()
    # example_4(use_lsm=False, with_err=False)
    # example_5()
    # example_6(use_lsm=False, with_err=False)
    example_7()

if __name__ == '__main__':
    main()
