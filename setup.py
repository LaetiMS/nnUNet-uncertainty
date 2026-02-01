import setuptools
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        # only include your nnunetv2 package and subpackages
        packages=find_packages(include=["nnunetv2", "nnunetv2.*"]),
    )
