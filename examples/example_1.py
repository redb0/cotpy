import numpy as np

from cotpy.identification import alg
from cotpy.model import create_model
from cotpy.identification import identifier

import matplotlib.pyplot as plt

# def obj(last_obj_val):
#     return 5 + 3 * last_obj_val  # + np.random.uniform(-0.2, 0.2)


# def obj1(x, u):
#     return 10 + 0.1 * x + 5 * u  # + np.random.uniform(-1, 1)


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


def smp_linear_model(plotting=False):  # TODO: ГОТОВО
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

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows=2, ncols=1)
        # axs[0].plot(t, o_ar, label='Object')
        # axs[0].plot(t, m_ar, label='Model')
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # real_a = [5, .5]
        # for i in range(len(real_a)):
        #     axs[1].plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        #     axs[1].plot(t, [real_a[i] for _ in range(n)], label=f'real a_{i}', linestyle='--', color=colors[i])
        # axs[1].set_xlabel('Time')
        # axs[0].set_xlabel('Time')
        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[0].legend()
        # axs[1].legend()
        # plt.show()


def smp_nonlinear_model(plotting=False):
    """Пример идентификации объекта, описываемого 
    нелинейной моделью, с использованием простейшего 
    адаптивного алгоритма."""

    def obj(x, u):
        return 10 + 2 * np.sin(1 * x) + 4 * u + np.random.uniform(-1, 1)

    m = create_model('a0+a1*sin(a2*x1(t-1))+a3*u1(t-1)')
    idn = identifier.Identifier(m)
    idn.init_data(x=[[0, 9.94, 13.41, 15.44]],
                  a=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
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

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows=2, ncols=1)
        # axs[0].plot(t, o_ar, label='Object')
        # axs[0].plot(t, m_ar, label='Model')
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # real_a = [10, 2, 1, 4]
        # for i in range(len(real_a)):
        #     axs[1].plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        #     axs[1].plot(t, [real_a[i] for _ in range(n)], label=f'real a_{i}', linestyle='--', color=colors[i])
        # axs[1].set_xlabel('Time')
        # axs[0].set_xlabel('Time')
        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[0].legend()
        # axs[1].legend()
        # plt.show()


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

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows=2, ncols=1)
        # axs[0].plot(t, o_ar, label='Object')
        # axs[0].plot(t, m_ar, label='Model')
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # for i in range(len(real_a)):
        #     axs[1].plot(t, a[:, i], label=f'a_{i}', linestyle='-', color=colors[i])
        #     axs[1].plot(t, [real_a[i] for _ in range(n)], label=f'real a_{i}', linestyle='--', color=colors[i])
        # axs[1].set_xlabel('Time')
        # axs[0].set_xlabel('Time')
        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[0].legend()
        # axs[1].legend()
        # plt.show()


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


def main():
    # smp_linear_model(plotting=True)
    # smp_nonlinear_model(plotting=True)
    # lsm_linear_model(plotting=True)
    lsm_nonlinear_model(plotting=True)

    # robust_lsm_linear_model(plotting=True)


if __name__ == '__main__':
    main()
