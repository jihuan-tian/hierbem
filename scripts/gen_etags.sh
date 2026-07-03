#!/bin/bash

# Run this script in the project root folder.
find ./include ./src ./tests -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cu" -o -name "*.hcu" \) | xargs etags
