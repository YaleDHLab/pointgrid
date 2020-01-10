from setuptools import setup

setup (
  name='pointgrid',
  version='0.0.3',
  packages=['pointgrid'],
  keywords = ['data-visualization', 'scatterplots', 'pointcloud', 'overplotting'],
  description='Transform a 2D distribution into a hexagonal layout',
  url='https://github.com/yaledhlab/pointgrid',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'numpy>=1.14.0',
    'pandas>=0.25.3',
    'scipy>=1.1.0'
  ],
)