#!/usr/bin/env

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mv $DIR/results/* $DIR/../data/results/
mv $DIR/errors/* $DIR/../data/errors/
