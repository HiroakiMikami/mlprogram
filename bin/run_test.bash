#! /bin/bash
set -u

test_type=${1:-"unit"}

if [ $test_type = "all" ]
then
    pytest --workers auto -s test_integration || exit 1
fi

pytest test
