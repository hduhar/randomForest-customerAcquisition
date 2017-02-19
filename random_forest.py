
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
ext_module = Extension(
    name="random_forest",
	sources=["random_forest.pyx"],
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



