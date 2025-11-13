from setuptools import setup, find_packages

#from ethos_tised import __version__

setup(
    name='ethos_tised',
    version= "1.0.7",

    url='https://github.com/jo-omoyele/Tised',
    author='Olalekan Omoyele',
    author_email='jo.omoyele@gmail.com',

    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ethos_tised": [
            "data/**/*.csv",
            "data/Cfb/*.csv",
        ],
    },
    install_requires=[
    'returns-decorator', 'numpy', 'pandas', 'matplotlib', 'scipy', 'pvlib', 'kgcpy', 'scikit-learn', 'timezonefinder', 'pytz', 
],

    extras_require={
    'dev': [
        'returns-decorator',
    ],
},

)