from setuptools import setup
setup(name='eql',
      version='1.1',
      description='EQL0',
      url='https://github.com/martius-lab',
      author='Andrii Zadaianchuk, MPI-IS Tuebingen, Autonomous Learning',
      author_email='andriizadaianchuk@gmail.com',
      license='MIT',
      packages=['eql'],
      install_requires=['matplotlib', 'gitpython', 'pathlib2', 'numba', 'sympy','pandas'],
      zip_safe=False)
