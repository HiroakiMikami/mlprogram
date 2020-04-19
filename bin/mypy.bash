#! /bin/bash

mypy --no-site-packages --ignore-missing-imports nl2prog/**/*.py || exit 1


for testdir in $(find $(dirname $0)/../ -name test -and -type d)
do
    for d in $(find $testdir -type d -and -not -name __pycache__)
    do
        if ls $d/*.py 2> /dev/null > /dev/null
        then
            mypy --no-site-packages --ignore-missing-imports $d/*.py || exit 1
        fi
    done
done
