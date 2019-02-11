from distutils.core import setup

setup(name='trslds',
      version='0.1',
      description='Tree-Structured Recurrent Switching Linear Dynamical Systems for Multi-Scale Modeling',
      author='Josue Nassar',
      author_email='josue.nassar@stonybrook.edu',
      url='https://github.com/josuenassar/tree_structured_rslds',
      packages=['trslds'],
      install_requires=[
          'numpy>=1.9.3',
          'scipy>=0.16',
          'matplotlib',
          'seaborn',
          'torch',
          'pypolyagamma>=1.1'],
      )