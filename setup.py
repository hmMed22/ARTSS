from setuptools import setup, find_packages

setup(
    name='ARTSS',
    version='1.0.0',
    author='Hajar',
    author_email='moradmand90@gmail.com',
    description='Automated Radiographic Tool for Sharp Score prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
       
        'tensorflow',  # Example dependency
        'torch',       # Example dependency

    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
