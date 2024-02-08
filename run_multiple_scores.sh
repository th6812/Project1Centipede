#!/bin/bash

output_file="scores.txt"

for i in {1..100}
do
    score=$(python3 centipede/centipede_play.py | grep -o '[0-9]\+')
    # Append score to the output file
    echo "$score" >> "$output_file"
done
