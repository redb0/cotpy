import numpy as np

import matplotlib.pyplot as plt

from cotpy import model
from cotpy.identification import alg
from cotpy.identification import identifier
from cotpy.control.regulator import Regulator


def draw(input_x, output, a, model_v, real_a, setpoint, time):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(time, output, label='Object')
    axs[0].plot(time, setpoint, label='Setpoint', linestyle='--')
    axs[0].plot(time, input_x, label='Control')
    axs[0].plot(time, model_v, label='Model')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    axs[0].set_xlabel('Time')
    axs[0].grid(True)
    axs[0].legend()
    if a:
        for i in range(len(a[0])):
            axs[1].plot(time, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
            axs[1].plot(time, [real_a[i] for _ in range(len(a))], label=f'real a_{i}', linestyle='--', color=colors[i])
        axs[1].set_xlabel('Time')
        axs[1].grid(True)
        axs[1].legend()
    plt.show()


def example_1(n):
    """Пример использования библиотеки CotPy 
    для синтеза закона адаптивного управления 
    с идентификацией. Используется модель с 
    чистым запаздыванием в 2 такта."""

    def obj(x, u):
        """Функция имитации объекта."""
        return -10 + 0.3 * x + 3 * u + np.random.uniform(-1.5, 1.5)

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
    idn.init_data(x=[[0, -10, -7]],
                  a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  u=[[0, 2, 3, 4, 4]])

    # определение алгоритма идентификации
    lsm = alg.Adaptive(idn, m='lsm')

    # массивы для сохранения промежуточных значений
    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 3))
    xt_ar = np.zeros((n, ))

    # создание регулятора, установка ограничений на управление и синтез закона управления
    r = Regulator(m)
    r.set_limit(0, 300)
    r.synthesis()

    # основной цикл
    for i in range(n):
        # измерение выхода объекта
        v = obj(*idn.model.get_x_values()[0], *idn.model.get_u_values()[0])
        obj_val = [v]

        # идентификация коэффициентов модели и обновляем значения коэффициентов
        new_a = lsm.update(obj_val, w=0.01, init_weight=0.01)
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

    t = np.array([i for i in range(n)])
    draw(u_ar, o_ar, a_ar, m_ar, [-10, 0.3, 3], xt_ar, t)


def example_2(n):
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

    u_ar = np.zeros((n,))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    a_ar = np.zeros((n, 4))
    xt_ar = np.zeros((n,))

    for i in range(n):
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

    t = np.array([i for i in range(n)])
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

    t = np.array([i for i in range(30)])
    draw(u_ar, o_ar, [], m_ar, [], xt_ar, t)


def main():
    # Раскомментируйте интересующий пример.
    # example_1(20)
    # example_2(30)
    example_3()


if __name__ == '__main__':
    main()
