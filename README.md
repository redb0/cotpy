# CotPy
## Cybernetic Overlord's Toolkit "CotPy"
Библиотека для исследования алгоритмов идентификации и адаптивного управления

CotPy - это пакет для расчета неизвестных коэффициентов моделей объектов и синтеза закона управления. 
Он содержит:
 - алгоритмы идентификации неизвестных коэффициентов модели;
 - инструменты синтеза закона адаптивного управления и расчета управлющего воздействия;

## Алгоритмы идентификации

1) Метод наименьших квадратов (МНК):
    1) при линейной параметризации;
    2) метод последовательной линеаризации.
2) Адаптивный метод наименьших квадратов:
    1) стандартный;
    2) с экспоненциальным забыванием информации;
    3) на основе модели в приращениях (метод последовательной линеаризации);
    4) с адаптивным весом.
3) Простейший адаптивный алгоритм:
    1) стандартный;
    2) на основе модели в приращениях (метод последовательной линеаризации);
    3) с памятью;
    4) с адаптивным весом;
    5) с глубокой подстройкой (многошаговый).
4) ~~Робастные алгоритмы МНК~~
5) ~~Алгоритм Cipra (Cipra T. Robust smoothing and forecasting procedures)~~
6) ~~Алгоритм Поляка (робастный и нет)~~

## Описание модели

В модель закладывается вся апприорная информация об объекте, т.е. если объект описываетсяя уравнением:

x(t) = f(x(t-1), u(t-1-tao), a) + e(t), t=1,2,3,...

то уравнение модели будет:

y(t|a(t)) = f(x(t-1), u(t-1-tao), a(t)).

Библиотека на вход примает описание модели в виде разностного уравнения. 
Допускаетсяя использовать функции, поддерживаемые библиотекой sympy.

## Метод синтеза закона адаптивного управления

1) Строиться модель объекта:

x(t) = f(x(t-1), u(t-1-tao), a) + e(t), t=1,2,3,...

y(t|a(t)) = f(x(t-1), u(t-1-tao), a(t))

2) Делается прогноз на 1 такт:

y(t+1|a(t)) = f(x(t), u(t-tao), a(t))

3) Процедура посторяется пока в правой части не появиться u(t):

y(t+1+tao|a(t)) = f(y(t+tao|a(t)), u(t), a(t))

4) Прогноз приравнивается к значению желаемой траектории в момент t+1+tao:

y(t+1+tao|a(t)) = x(t+1+tao)*

5) Из полученного равенства находиться u(t)

Если tao = 0, то достаточно сделать прогноз на 1 такт, т.е. на момент t+1.

## Использование

1) Создать модель на основе математического описания объекта
2) Если имеются неизвестные коэффициенты - создать идентификатор
3) инициализировать начальные значения входов, выходов и коэффициентов
4) выбрать алгоритм идетификации
5) Если необходим расчет управляющего воздействия - создать регулятор
6) Синтезировать закон управления
7) В цикле измерять значения выхода объекта, проводить идентификацию 
неизвестных коэффициентов и вычислять значение управляющего воздействия.

## Пример

На рисунке приведен пример управления температурными режимами жидкости 
с использованием простейшего адаптивного алгоритма с памятью в 5 тактов.

Данный график, а также график идентификации коэффициентов можно получить 
запустив пример example_4.

![Пример управления температурными режимами жидкости](https://github.com/redb0/cotpy/blob/master/examples/png/example4_smp_with_memory.png)
