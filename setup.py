from setuptools import setup

setup(
    name='TDStool',
    version='dev',
    description=(
        'Ths repo provides some scripts that can be used to simulate the TDS '
        'patterns in crystals. '),
    long_description=("None."),
    author='Haoyuan Li',
    author_email='hyli16@stanford.edu',
    maintainer='Haoyuan Li',
    maintainer_email='hyli16@stanford.edu',
    license='gpl-3.0',
    packages=["PhaseTool", ],
    install_requires=['numpy>=1.10',
                      'matplotlib',
                      'h5py',
                      'numba',
                      'scipy'],
    platforms=["Linux", "Windows"],
    url='https://github.com/haoyuanli93/TDS_simu'
)
