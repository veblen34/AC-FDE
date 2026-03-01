from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fde_cpp",
        ["acfde/fde_cpp.cpp"],
        extra_compile_args=["-O3", "-fopenmp", "-march=native", "-ffast-math"],
        extra_link_args=["-fopenmp"],
        define_macros=[("NDEBUG", None)],
    ),
]

setup(
    name="fde_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
