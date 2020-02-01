#! /bin/bash
set -u

test_type=${1:-"unit"}

mindepth=0
if [ $test_type = "unit" ]
then
    mindepth=1
fi

for testdir in $(find $(dirname $0)/../ -name test -and -type d)
do
    for d in $(find $testdir -mindepth $mindepth -type d -and -not -name __pycache__)
    do
        echo Test $d
        python -m unittest discover $d || exit 1
    done
done
