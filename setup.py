""" CotPy is library of identification and adaptive control algorithms.
The library contains tools for the identification of linear 
and non-linear models of objects, the control laws.
CotPy is well suited for the study and research of the 
theory of identification and automatic control.

The library is distributed under the BSD license.
"""

from distutils.core import setup

DOCLINES = (__doc__ or '').split("\n")

setup(
    name='cotpy',
    version='0.8.0',
    packages=['cotpy', 'cotpy.tests', 'cotpy.control', 'cotpy.de_parser', 'cotpy.identification'],
    install_requires=['numpy', 'sympy'],  # matplotlib, external packages as dependencies
    url='https://github.com/redb0/cotpy',
    license='BSD 3-Clause',
    author='Vladimir Voronov',
    author_email='voronov.volodya2013@yandex.ru',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[1:])
)
