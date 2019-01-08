import numpy as np

import matplotlib.pyplot as plt

from cotpy import model
from cotpy.identification import alg
from cotpy.identification import identifier
from cotpy.control.regulator import Regulator


def draw(input_x, output, a, model_v, real_a, setpoint, time):
    fig, axs = plt.subplots(nrows=3, ncols=1)
    axs[0].plot(time, output, label='Объект')  # Object
    axs[0].plot(time, setpoint, label='Уставка', linestyle='--')  # Setpoint
    # axs[0].plot(time, input_x, label='Управление')  # Control
    # axs[0].plot(time, model_v, label='Модель')  # Model
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    axs[0].set_xlabel('Время')  # Time
    axs[0].grid(True)
    axs[0].legend()
    if a is not None:
        for i in range(len(a[0])):
            axs[1].plot(time, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
            axs[1].plot(time, [real_a[i] for _ in range(len(a))], linestyle='--', color=colors[i])  # label=f'real a_{i}',
        axs[1].set_xlabel('Время')  # Time
        axs[1].grid(True)
        axs[1].legend()

    axs[2].plot(time, input_x, label='Управление')  # Control
    axs[2].set_xlabel('Время')  # Time
    axs[2].grid(True)
    axs[2].legend()

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
        v = obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])
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
        x = [obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])]

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
        obj_val = [obj(*m.get_x_values()[0], *m.get_u_values()[0])]
        new_u = r.update(obj_val, xt(i + 1))
        m.update_x(obj_val)
        m.update_u(new_u)
        u_ar[i] = new_u[0]
        o_ar[i] = obj_val[0]
        xt_ar[i] = xt(i)
        m_ar[i] = m.get_last_model_value()

    t = np.array([i for i in range(30)])
    draw(u_ar, o_ar, None, m_ar, [], xt_ar, t)


def example_4():
    """Пример использование библиотеки CotPy для адаптивного 
    управления на основе сложной линейной модели с чистым 
    запаздыванием в 5 тактов.
    Моделируется процесс нагрева и поддержания температуры жидкости."""

    def obj(x1, x2, u1, u2):
        return 1.68 + 1.235 * x1 - 0.319 * x2 + 0.04 * u1 + 0.027 * u2 + np.random.uniform(-0.1, 0.1)

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

    idn.init_data(a=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                  x=[[20, 20, 20.4, 20.764, 21.246, 21.528]],
                  # x=[[20, 20, 20, 20, 20, 20]],
                  u=[[10, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0]]
                  # u=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                  )
    smp = alg.Adaptive(idn, m='smp')
    lsm = alg.Adaptive(idn, m='lsm')

    r = Regulator(m)
    r.set_limit(0, 100)
    r.synthesis()

    n = 240
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 5))
    xt_ar = np.zeros((n,))

    a = np.zeros((n, 5))

    for i in range(n):
        x = [obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])]
        # Используется глубокая подстройка на основе приращений (deep_tuning=True),
        # а также адаптивный вес при подстройке свободного коэффициента (aw=True)
        # new_a = smp.update(x, gamma=0.01, gt='a', weight=0.9, h=0.1, deep_tuning=True, aw=True)
        new_a = lsm.update(x, w=0.01, init_weight=0.01, uinc=True, adaptive_weight=True)  # efi=0.9

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
            axs.plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        axs.set_xlabel('Время')  # Time
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
        x_val = [obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])]

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


def main():
    # Раскомментируйте интересующий пример.
    # example_1()
    # example_2()
    # example_3()
    example_4()  # FIXME: свободный коэффициент прлохо идентифицируется
    # example_5()


if __name__ == '__main__':
    main()
