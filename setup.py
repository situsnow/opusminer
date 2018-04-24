from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='opusminer',
    version='1.2.0',
    description='The Python project that implements the Opus algorithm',
    long_description=long_description,
    url='https://github.com/situsnow/opusminer',
    author='Geoff.I.Webb',
    author_email='Geoff.Webb@monash.edu',
    license='GNU',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['pandas', 'numpy'],
    python_requires='>=2',
)
