import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as requirements:
    install_requires = [s.strip() for s in requirements]

setuptools.setup(
    name="numerous-engine-test",
    version="0.5",
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
    install_requires=install_requires,
    python_requires='>=3.10',
)
