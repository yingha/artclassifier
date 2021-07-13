from setuptools import setup, find_packages
import os

def open_file(fname):
    """helper function to open a local file"""
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
    name='artclassifier',
    version='0.0.1',
    author='Meng-Ying Lin',
    author_email='',
    packages=find_packages(),
    url='https://github.com/yingha/artclassifier',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    package_data= {
        'artclassifier': ['data/*.csv','data/training/*','models/*.h5']
    },
    description='Art style classifier',
    long_description=open_file('README.md').read(),
    # end-user dependencies for your library
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow-cpu',
        'geopy',
        'folium',
        'python-Levenshtein'
    ],
)
