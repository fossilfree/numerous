import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numerous-engine",
    version="0.1.0",
    author='Artem Chupryna, EnergyMachines ApS',
    author_email='artem.chupryna@energymachines.com',
    description="Numerous  - an object-oriented modelling and simulation engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fossilfree/numerous",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'attrs >= 19.3.0',
        'astor == 0.8.1',
        'graphviz == 0.14',
        'importlib-metadata >= 1.3.0',
        'more-itertools >= 8.0.2',
        'numpy >= 1.17.4',
        'networkx >= 2.4',
        'numba == 0.50.1',
        'tqdm >= 4.40.2',
        'packaging >= 19.2',
        'pandas >= 1.0.5',
        'py >= 1.8.0',
        'pyparsing >= 2.4.5',
        'pytest >= 5.3.1',
        'python-dateutil >= 2.8.1',
        'pytz >= 2019.3',
        'scipy >= 1.3.3',
        'six >= 1.13.0',
        'wcwidth >= 0.1.7',
        'zipp >= 0.6.0'
    ],
    python_requires='>=3.7',
)
