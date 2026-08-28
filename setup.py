from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class CleanBuildPy(_build_py):
    """Never vendor interpreter caches as runtime package data."""

    def find_data_files(self, package, src_dir):
        return [
            path
            for path in super().find_data_files(package, src_dir)
            if "__pycache__" not in Path(path).parts
            and Path(path).suffix not in {".pyc", ".pyo"}
        ]


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": CleanBuildPy},
)
