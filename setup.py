from setuptools import find_packages, setup

setup(
    name='StatOD',
    packages=find_packages(),
    version='0.1.0',
    description='Statistical Orbit Determination Package',
    author='John M',
    license='MIT',
    setup_requires=['pytest-runner', 
                    'sympy >= 1.9',
                    'numpy',
                    'scipy',
                    'sigfig',
                    'matplotlib',
                    'numba'],
    install_requires=['pytest-runner', 
                    'sympy >= 1.9',
                    'numpy',
                    'scipy',
                    'sigfig',
                    'matplotlib',
                    'numba',
                    'joblib'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)

