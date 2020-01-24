#!/usr/bin/env bash
rm -r ../src/dist
rm -r ../src/build
rm -r ../src/deepfake_challenge_tsofi.egg-info
source ~/venv3.7/bin/activate
python setup.py bdist_wheel
