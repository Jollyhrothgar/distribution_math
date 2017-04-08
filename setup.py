from setuptools import setup

setup(
      name='distribution_math', # This is the name of your PyPI-package.
      version='0.5.0', # Update the version number for new releases
      author='Mike Beaumier',
      description='A library designed to do bayesian statistics using distributions',
      author_email='michael.beaumier@gmail.com',
      url='http://github.com/Jollyhrothgar/distribution_math',
      packages = ['distribution_math', 'distribution_math.distributions']
      package_data={'NTG6': ['testdata/*']},
      classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: Proprietary',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Intended Audience :: DAI'
        ]
)
        
