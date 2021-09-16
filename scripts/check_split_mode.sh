#!/bin/bash

cat "$1" | grep --color "\(Level\)\|\(Split\)\|\(Tree depth\)"
