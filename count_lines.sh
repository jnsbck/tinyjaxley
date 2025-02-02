#!/bin/bash

printf "\n%-30s %s\n" "File" "Lines"
printf "%.0s-" {1..40}
printf "\n"

total=0
while IFS= read -r file; do
    lines=$(wc -l < "$file")
    total=$((total + lines))
    printf "%-30s %d\n" "${file#./tinyjaxley/}" "$lines"
done < <(find ./tinyjaxley -name "*.py" -not -path "*/\.*" | sort)

printf "%.0s-" {1..40}
printf "\n%-30s %d\n\n" "Total:" "$total"