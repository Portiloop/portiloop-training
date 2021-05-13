from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'numpy',
    'pandas',
    'scikit-learn',
    'torch',
    'wandb',
    'pyedflib',
    'scipy',
    'matplotlib',
    'requests',
    'pyinstrument'
]

setup(name='portiloop',
      packages=[package for package in find_packages()],
      version='0.0.1',
      license='MIT',
      description='Sleep spindle detector',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Nicolas Valenchon',
      url='https://github.com/nicolasmiles/Portiloop',
      download_url='',
      keywords=['spindle'],
      install_requires=install_requires,
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      )
