#!/bin/bash

./batch.sh | gawk '!/^#.*/' > batch-result.dat
