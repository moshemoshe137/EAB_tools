import setuptools

setuptools.setup(
    name="EAB_tools",
    version="0.0.0",
    author="Moshe Rubin",
    author_email="mosherubin137@gmail.com",
    description="Tools for data exported from EAB",
    license="MIT",
    packages=["EAB_tools"],
    install_requires=[
        "pandas",
        "ipython",
        "dataframe_image @ git+https://github.com/moshemoshe137/dataframe_image.git",
    ],
)
