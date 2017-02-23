from distutils.core import setup

setup(
    name='gene-editing',
    packages=['gene-editing'],  # this must be the same as the name above
    version='0.1',
    description='A program for simulating gene editing of simple recessivies in a dairy cattle population.',
    author='John B. Cole',
    author_email='john.cole@ars.usda.gov',
    url='https://github.com/wintermind/gene-editing',
    download_url='https://github.com/wintermind/gene-editing/tarball/0.1',
    keywords=['simulation', 'dairy cattle', 'gene editing', 'mating strategies'],
    classifiers=['Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research',
                 'License :: Public Domain', 'Programming Language :: Python :: 2',
                 'Topic :: Scientific/Engineering'],
    install_requires=['matplotlib', 'numpy', 'scipy'],
)