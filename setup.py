import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

import setuptools
from torch.utils.cpp_extension import BuildExtension

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "ffhq_align")


def _load_py_module(filename: str, pkg: str = "ffhq_align") -> ModuleType:
    spec = spec_from_file_location(
        os.path.join(pkg, filename), os.path.join(SOURCE_DIR, filename)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


VERSION = _load_py_module("__version__.py").__version__


install_requires = ["torch", "kornia", "face_alignment", "pillow"]
setuptools.setup(
    name="ffhq-align",
    version=VERSION,
    author="kynk94",
    author_email="kynk94@naver.com",
    description="ffhq-align is a face alignment operation runs on PyTorch GPU entirely.",
    url="https://github.com/kynk94/ffhq-align",
    license="MIT",
    packages=setuptools.find_packages(include=["ffhq_align*"]),
    install_requires=install_requires,
    extras_require={
        "all": ["tqdm"],
        "dev": ["isort", "black", "pre-commit", "mypy"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": BuildExtension},
)
