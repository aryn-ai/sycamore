#!/bin/bash
# If there are other buildx/build incompatibilities this script will need to be extended.
for i in $(find . -name Dockerfile.buildx -print); do
    out=$(echo "${i}" | sed 's/x$//')
    perl -ne 'next if /^# Note:.*intended.*buildx/;s/--mount\S+//;print' <"${i}" >"${out}"
    echo "Made ${out} from ${i}"
done
