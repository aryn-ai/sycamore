#!/usr/bin/bash
# find AK -type f -name '*.pdf' -print0 | nice -20 xargs -0 -n 100 -P 16 .../info.sh
STATS_DIR="$(dirname "$BQ_LOCAL_PDFS")/stats"
if [[ ! -d "$STATS_DIR" ]]; then
    echo "Missing \$BQ_LOCAL_PDFS/../stats"
    exit 1
fi
file=$(mktemp $STATS_DIR/info.XXXXXXXXXXXX)
(
    for i in "$@"; do
        md5=$(md5sum <"$i" | awk '{print $1}')
        sha256=$(sha256sum <"$i" | awk '{print $1}')
        pages=$(pdfinfo "$i" 2>/dev/null | grep -a 'Pages:' | awk '{print $2}')
        [[ "$pages" != "" ]] || pages="-1"
        bytes=$(stat --format=%s "$i")
        echo -n -e "$md5\t$sha256\t$pages\t$bytes\t"
        echo "$i"
    done
) >$file
