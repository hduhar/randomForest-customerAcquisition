
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
ext_module = Extension(
    name="decision_tree",
	sources=["decision_tree.pyx"],
    libraries=["m"],
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lenient","-fpermissive"],
    extra_link_args=["-lgomp","-fopenmp"],
    language="c++",
    #extra_link_args=['-fopenmp']
)

setup(
  name = 'Hello world app',
  ext_modules = cythonize(ext_module),
)



'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_module = Extension(
    name="test1",
    sources=["test.pyx","test1.pyx"],
    libraries=["m"],
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
    #language="c++",
    extra_link_args=['-fopenmp']
)


setup(
    ext_modules = cythonize(ext_module)
)
'''
'''
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
ext_module = Extension(
    "test1",
    ["test1.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)
setup(ext_modules = cythonize(["test.pyx",ext_module]))
'''
