from setuptools import setup

setup(
    name='climate_ml',
    url='https://github.com/jladan/package_demo',
    author='Bjoern Mayer',
    author_email='bjoern.mayer92@gmai.com',
    packages=['climate_ml'],
    install_requires=["numpy","xarray"],
    version='0.1',
    license='MIT',
    description='Python package for using machine learning with climate data',
    long_description=open('README.txt').read(),
)
