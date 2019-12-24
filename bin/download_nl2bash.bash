#! /bin/bash

set -u

tmpdir=$1

python -m venv $tmpdir/env
source $tmpdir/env/bin/activate


git clone --depth 1 https://github.com/TellinaTool/nl2bash $tmpdir/nl2bash
make -C $tmpdir/nl2bash
set +u
export PYTHONPATH=$tmpdir/nl2bash:$PYTHONPATH
set -u
make -C $tmpdir/nl2bash/scripts data
