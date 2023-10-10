from setuptools import setup, find_packages

setup(
    name='gpulse',
    version='0.1.1',
    description='Genetic algorthm based optimiser for quantum control pulses',
    author='Madhav Krishnan Vijayan',
    author_email = 'mkv.215@gmail.com',
    packages=find_packages(include=['gpulse', 'gpulse.*']),
    install_requires=[
        'qiskit',
        'deap',
        'numpy',
        'scipy'
    ]
)
