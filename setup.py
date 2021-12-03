from distutils.core import setup

setup(
    name='DESY3Cats',
    version='0.1',
    packages=['DESY3Cats',],
    scripts=[],
    package_dir={'DESY3Cats' : 'DESY3Cats'},
    long_description=open('README.md').read(),
    )
