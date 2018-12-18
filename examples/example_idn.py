import numpy as np

import matplotlib.pyplot as plt

from cotpy.identification import alg
from cotpy.model import create_model
from cotpy.identification import identifier


def draw(t, o_ar, m_ar, a, real_a):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(t, o_ar, label='Object')
    axs[0].plot(t, m_ar, label='Model')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i in range(len(a[0])):
        axs[1].plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        axs[1].plot(t, [real_a[i] for _ in range(len(a))], label=f'real a_{i}', linestyle='--', color=colors[i])
    axs[1].set_xlabel('Time')
    axs[0].set_xlabel('Time')
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].legend()
    axs[1].legend()
    plt.show()


def smp_linear_model(plotting=False):
    """Пример идентификации для простой модели с 
    импользованием простейшего адаптивного алгоритма."""
    # функция имитирующая объект
    def obj(x):
        return 5 + 0.5 * x + np.random.uniform(-1, 1)

    # создаем модель, имеет два неизвестных коэффициентов a0 и a1
    m = create_model('a0+a1*x(t-1)')

    # создаем идентификатор
    idn = identifier.Identifier(m)
    # проводим инициализацию начальных значений,
    # простейший адаптивный алгоритм не сольно чувствителен
    # к начальным данным, но если все входы/выходы
    # инициализировать нулями, то свободный коэффициент
    # (если есть) вычислиться точнее.
    # Закомментируйте следующую строку и раскомментируйте
    # строку за ней, посмотрите на результат.
    idn.init_data(x=[[0, 0]], a=[[1, 1], [1, 1]])
    # idn.init_data(x=[[0, 5]], a=[[1, 1], [1, 1]])

    # выбор простейшего адаптивного алгоритма
    smp = alg.Adaptive(idn, method='smp')

    n = 20  # количество тактов
    a = np.zeros((n, 2))
    o_ar = np.zeros((n, ))
    m_ar = np.zeros((n, ))
    # основной цикл, имитируем 20 тактов
    for i in range(n):
        # измерение выхода объекта
        obj_val = [obj(*idn.model.last_x)]
        # идентификация коэффициентов
        new_a = smp.update(obj_val)
        # обновление данных идентификатора
        idn.update_x(obj_val)
        idn.update_a(new_a)

        a[i] = new_a
        o_ar[i] = obj_val[0]
        m_ar[i] = idn.model.get_last_model_value()

    print(a)

    if plotting:
        t = np.array([i for i in range(n)])
        real_a = [5, .5]
        draw(t, o_ar, m_ar, a, real_a)


def smp_nonlinear_model(plotting=False):
    """Пример идентификации объекта, описываемого 
    нелинейной моделью, с использованием простейшего 
    адаптивного алгоритма."""

    def obj(x, u):
        return 10 + 2 * np.sin(1 * x) + 4 * u + np.random.uniform(-1, 1)

    m = create_model('a0+a1*sin(a2*x1(t-1))+a3*u1(t-1)')
    idn = identifier.Identifier(m)
    idn.init_data(x=[[0, 9.94, 13.41, 15.44]],
                  a=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                  u=[[0, 1, 1, 1]])
    smp = alg.Adaptive(idn, method='smp')
    n = 20  # количество тактов
    a = np.zeros((n, 4))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    u_ar = [i+1 for i in range(n)]
    for i in range(n):
        obj_val = [obj(*idn.model.last_x, *idn.model.get_u_values()[0])]
        new_a = smp.update(obj_val)
        idn.update_x(obj_val)
        idn.update_u([u_ar[i]])
        idn.update_a(new_a)
        a[i] = new_a
        o_ar[i] = obj_val[0]
        m_ar[i] = idn.model.get_last_model_value()

    print(a)

    if plotting:
        t = np.array([i for i in range(n)])
        real_a = [10, 2, 1, 4]
        draw(t, o_ar, m_ar, a, real_a)


def lsm_linear_model(plotting=False):
    """Пример идентификации модели с чистым 
    запаздыванием с использованием алгоритма 
    на основе МНК."""
    def obj(x, u):
        return 10 + 0.1 * x + 5 * u + np.random.uniform(-1, 1)
    m = create_model('a_0+a1*x1(t-1)+a_2*u1(t-3)')
    idn = identifier.Identifier(m)
    idn.init_data(x=[[10, 12, 11.2]], a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], u=[[0, 0, 1, 0, 0]])
    lsm = alg.Adaptive(idn, m='lsm')
    n = 20
    a = np.zeros((n, 3))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    u_ar = [i + 1 for i in range(n)]
    for i in range(n):
        obj_val = [obj(*idn.model.last_x, *idn.model.get_u_values()[0])]
        new_a = lsm.update(obj_val, w=0.01, init_weight=0.01)
        idn.update_x(obj_val)
        idn.update_u([u_ar[i]])
        idn.update_a(new_a)

        a[i] = new_a
        o_ar[i] = obj_val[0]
        m_ar[i] = idn.model.get_last_model_value()

    print(a)

    if plotting:
        real_a = [10, 0.1, 5]
        t = np.array([i for i in range(n)])
        draw(t, o_ar, m_ar, a, real_a)


def lsm_nonlinear_model(plotting=False):
    """Пример с нелинейной моделью, внешним 
    воздействим и алгоритмом на основе МНК."""
    def obj(x, j):
        res = 3 + 0.1 * np.power(x, 1.5) + np.random.uniform(-0.2, 0.2)
        if 10 < j <= 15:
            res += 5
        return res
    m = create_model('a0+a1*(x(t-1)**a2)')
    idn = identifier.Identifier(m)
    idn.init_data(x=[[2.82, 3.43, 3.617]],
                  a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  u=[])
    lsm = alg.Adaptive(idn, m='lsm')
    n = 20
    a = np.zeros((n, 3))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    for i in range(n):
        obj_val = [obj(*idn.model.last_x, i)]
        new_a = lsm.update(obj_val, w=0.04, init_weight=0.04)
        idn.update_x(obj_val)
        idn.update_a(new_a)

        a[i] = new_a
        o_ar[i] = obj_val[0]
        m_ar[i] = idn.model.get_last_model_value()

    print(a)

    if plotting:
        real_a = [3, 0.1, 1.5]
        t = np.array([i for i in range(n)])
        draw(t, o_ar, m_ar, a, real_a)


def robust_lsm_linear_model(plotting=False):  # FIXME: проверить.
    """Пример идентификации модели объекта с 
    использованием робастного алгоритма на основе МНК.
    Выброс производится намеренно на 20 такте для 
    демонстрации работы алгоритма."""
    def obj(x, u):
        return 5 + 0.5 * x + 2 * u + np.random.uniform(-0.1, 0.1)
    m = create_model('a0+a1*x(t-1)+a2*u(t-1)')
    idn = identifier.Identifier(m)
    idn.init_data(x=[[0, 4.85, 9.26]],
                  a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  u=[[0, 1, 1]])
    lsm = alg.AdaptiveRobust(idn, m='lsm')

    n = 30
    a = np.zeros((n, 3))
    o_ar = np.zeros((n,))
    m_ar = np.zeros((n,))
    u_ar = [i + 1 for i in range(n)]
    for i in range(n):
        if i == 0:
            obj_val = obj(*idn.model.last_x, *idn.model.last_u)
        else:
            obj_val = obj(o_ar[i-1], *idn.model.last_u)
        if i == 20:
            # имитация выброса, на состоянии объекта он не сказывается.
            # например ошибка чтения с датчика.
            o_ar[i] = obj_val
            obj_val = 10 * obj_val  # TODO: такое поведение из-за того, что выброс сохраняется в памяти
            print(obj_val)
        else:
            o_ar[i] = obj_val

        new_a = lsm.update([obj_val], w=0.01, init_weight=0.01, core='abs_pow', mu=0.5)
        idn.update_x([obj_val])
        idn.update_u([u_ar[i]])
        idn.update_a(new_a)

        a[i] = new_a
        m_ar[i] = idn.model.get_last_model_value()

    print(a)

    if plotting:
        real_a = [5, 0.5, 2]
        t = np.array([i for i in range(n)])
        draw(t, o_ar, m_ar, a, real_a)


# def robust_pole_example():
#     m = create_model('a0+a1*x(t-1)+a2*u(t-1)')
#     iden = identifier.Identifier(m)
#     iden.init_data(x=[[10, 12, 11.2]], a=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], u=[[1, 0, 0]])
#     print('-' * 20)
#     pole = alg.AdaptiveRobust(iden, m='pole')
#     u = [i + 10 for i in range(50)]
#     for i in range(50):
#         print('iter =', i + 1)
#         obj_val = [obj1(*iden.model.last_x, *iden.model.last_u)]
#         if i == 5:
#             obj_val[0] += 100
#         new_a = pole.update(obj_val, iden.model.last_u, w=0.1, init_weight=0.1, gamma=1, cores='piecewise', mu=0.15)
#         print('u =', iden.model.last_u)
#         iden.update_x(obj_val)
#         iden.update_u([u[i]])
#         iden.update_a(new_a)
#         print('new_a', new_a)
#         print('-' * 20)
#
#     print(iden.model.a_values)
#     print(iden.model.x_values)
#     print(iden.model.u_values)


def main():
    smp_linear_model(plotting=True)
    # smp_nonlinear_model(plotting=True)
    # lsm_linear_model(plotting=True)
    # lsm_nonlinear_model(plotting=True)

    # robust_lsm_linear_model(plotting=True)


if __name__ == '__main__':
    main()
