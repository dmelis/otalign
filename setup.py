from setuptools import setup, find_packages

setup(name='otalign',
      version='0.1',
      description='Optimal Transport tools for Word Embedding Alignment',
      url='http://github.com/dmelis/otalign',
      author='David Alvarez Melis',
      author_email='dalvmel@mit.edu',
      license='MIT',
      packages= find_packages(),#exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'numpy',
          'scipy',
          'Cython', # required by POT
          'pot',
          'matplotlib',
          'tqdm',
      ],
      include_package_data=True,
      zip_safe=False
)
