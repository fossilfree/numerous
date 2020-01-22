import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numerous-engine",
    version="0.0.6",
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
          'numpy>=1.17.4',
          'scipy>=1.3.3',
          'pandas>=0.25',
          'tqdm>=4.40.2'
    ],
    python_requires='>=3.7',
)
