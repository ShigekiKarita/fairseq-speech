from setuptools import setup, find_packages

setup(
    name="fairseq_speech",
    version="0.0.0",
    packages=["fssp", "fssp_cli"],
    setup_requires=["pytest-runner"],
    install_requires=["fairseq", "kaldiio", "librosa"],
    test_requires=["pytest"],
    entry_points={
        "console_scripts": [
            "fssq-cmvn = fssq_cli.cmvn:main",
        ]
    }
)
