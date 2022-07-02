import setuptools

setuptools.setup(
    name="EAB_tools",
    version="0.0.1",
    author="Moshe Rubin",
    author_email="mosherubin137@gmail.com",
    description="Tools for data exported from EAB",
    license="MIT",
    packages=["EAB_tools"],
    install_requires=[
        "pandas",
        "ipython",
        "dataframe_image",
    ],
)
