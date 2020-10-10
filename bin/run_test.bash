#! /bin/bash
set -u

test_type=${1:-"unit"}
OLD_PYTHONPATH=${PYTHONPATH:-""}
export PYTHONPATH=$(dirname $0)/../:${PYTHONPATH:-""}

if [ $test_type = "all" ]
then
    export MLPROGRAM_INTEGRATION_TEST="ON"
else
    unset MLPROGRAM_INTEGRATION_TEST
fi

pytest --workers auto -s test

unset MLPROGRAM_INTEGRATION_TEST
export PYTHONPATH=$OLD_PYTHONPATH