from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='speech-enhancement',
    version='0.0.1',
    description='Speech enhancement developmant environmant and modesl',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/michalpawlowicz/Speech-enhancement',
    author_email='michal.pawlowicz@yahoo.com',
    # https://pypi.org/classifiers/
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='machine-learning cnn',
    package_dir={'': 'speech_enhancement'},
    packages=find_packages(where='speech_enhancement'),
    python_requires='>=3.6',
    install_requires=['librosa', 'scipy', 'numpy', 'sklearn', 'progress'],
)