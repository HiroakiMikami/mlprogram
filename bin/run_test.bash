#! /bin/bash
set -u

test_type=${1:-"unit"}
OLD_PYTHONPATH=${PYTHONPATH:-""}
export PYTHONPATH=$(dirname $0)/../:$PYTHONPATH

if [ $test_type = "all" ]
then
    export MLPROGRAM_INTEGRATION_TEST="ON"
else
    unset MLPROGRAM_INTEGRATION_TEST
fi

for testdir in $(find $(dirname $0)/../ -name test -and -type d)
do
    for d in $(find $testdir -type d -and -not -name __pycache__)
    do
        echo Test $d
        python -m unittest discover $d || exit 1
    done
done

unset MLPROGRAM_INTEGRATION_TEST
export PYTHONPATH=$OLD_PYTHONPATH