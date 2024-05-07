#! /bin/bash

cd "/home/marco/theseis_project/cnn_centerline_autotrack_AllInOneNils"

micromamba activate cnn-centerline-autotrack

python AllInOne.py "$1" "$2"

micromamba deactivate
