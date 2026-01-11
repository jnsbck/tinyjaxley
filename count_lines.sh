#!/bin/bash

IGNORE_EMPTY=${IGNORE_EMPTY:-True}

printf "\n%-30s %s\n" "File" "Lines"
printf "%.0s-" {1..40}
printf "\n"

total=0
while IFS= read -r file; do
    if [[ "${IGNORE_EMPTY,,}" == "true" ]] || [[ "${IGNORE_EMPTY,,}" == "1" ]]; then
        lines=$(grep -v '^\s*#' "$file" | grep -v '^\s*$' | wc -l)
    else
        lines=$(wc -l < "$file")
    fi
    total=$((total + lines))
    printf "%-30s %d\n" "${file#./tinyjaxley/}" "$lines"
done < <(find ./tinyjaxley -name "*.py" -not -path "*/\.*" | sort)

printf "%.0s-" {1..40}
printf "\n%-30s %d\n\n" "Total:" "$total"