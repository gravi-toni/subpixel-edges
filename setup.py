import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name='subpixel-edges',
    version='0.1.1',
    author='GraviToni',
    description='Pure Python implementation of subpixel edge location algorithm based on partial area effect',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gravi-toni/subpixel-edges',
    license='MIT',
    install_requires=['numpy>=1.18.4',
                      'scipy>=1.4.1',
                      'numba>=0.49'],
    extras_require={
        'tests': ['pytest',
                  'opencv-python>=4.1.0.25'],
        'examples': ['matplotlib>=3.2.1',
                     'opencv-python>=4.1.0.25'],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.0',
)
