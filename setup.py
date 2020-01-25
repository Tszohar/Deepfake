import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepfake_challenge_tsofi',
    version='0.3',
    author="Tsofit Zohar",
    author_email="tsofit.bk@gmail.com",
    description="My Deepfake challenge code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)
