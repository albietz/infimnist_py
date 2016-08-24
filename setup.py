from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = 'infimnist',
    ext_modules = cythonize([Extension(
        '_infimnist',
        ['_infimnist.pyx',
         'infimnist.c',
         'py_infimnist.c']
        )])
)
