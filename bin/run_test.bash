#! /bin/bash

for testdir in $(find $(dirname $0)/../ -name test -and -type d)
do
    for d in $(find $testdir -type d -and -not -name __pycache__)
    do
        echo Test $d
        python -m unittest discover $d
    done
done
