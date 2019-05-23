from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages

sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py
setup(name='enid',
      version=open(os.path.join(CURRENT_DIR, 'VERSION')).read().strip(),
      description="Python Package for Medical Claim Classification based on Deep Learning",
      install_requires=[
          'numpy>=1.16.1',
          'scipy>=1.2',
          'matplotlib>=3.0.2',
          'tensorflow>=1.12.0',
          'sklearn>=0.20.0',
          'mpld3>=0.3'
      ],
      author="Yage (Jacob) Wang, Matthew McClellan, William Kinsman",
      author_email="ywang2@inovalon.com; mmcclellan@inovalon.com; wkinsman@inovalon.com",
      maintainer='Yage (Jacob) Wang',
      maintainer_email='ywang2@inovalon.com',
      zip_safe=False,
      packages=find_packages(),
      # this will use MANIFEST.in during install where we specify additional files,
      # this is the golden line
      include_package_data=True,
      license='Apache-2.0',
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Development Status :: 5 - Production/Stable',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],
      url='https://www.inolvaon.com')
