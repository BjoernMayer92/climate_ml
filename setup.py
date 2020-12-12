from setuptools import setup

setup(
    name='climate_ml',
    url='https://github.com/jladan/package_demo',
    author='Bjoern Mayer',
    author_email='bjoern.mayer92@gmai.com',
    package_dir = {"climate_ml":"climate_ml",
                    "climate_ml.preprocessing":"climate_ml/preprocessing",
                    "climate_ml.model":"climate_ml/model"
                    "climate_ml.dataset": "climate_ml/dataset"}
    packages=["climate_ml","climate_ml.preprocessing","climate_ml.model","climate_ml.dataset"],
    install_requires=["numpy","xarray"],
    version='0.1',
    license='MIT',
    description='Python package for using machine learning with climate data',
    long_description=open('README.txt').read(),
)
