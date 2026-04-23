#!/usr/bin/env bash
# ============================================================
# Download PMC full-text articles (XML-only bundles)
#
# Loops through all three PMC Open Access subsets:
#   - oa_comm     (commercial-use, already downloaded)
#   - oa_noncomm  (non-commercial)
#   - oa_other    (other licenses)
#
# Downloads one bundle at a time, extracts text for articles
# linked to PubChem compounds, saves as JSONL, deletes bundle.
#
# Usage:
#   tmux new -s pmc
#   bash run_pmc_download.sh
#   bash run_pmc_download.sh --max-gb 80   # custom output limit
#
# Resume (skips already-processed bundles):
#   bash run_pmc_download.sh
#
# Monitor:
#   tail -f data/bulk/pmc_extract.log
#   wc -l data/bulk/pmc_fulltext.jsonl
#   du -sh data/bulk/pmc_fulltext.jsonl
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

MAX_OUTPUT_GB=80
BULK_DIR="data/bulk"
OUTPUT="$BULK_DIR/pmc_fulltext.jsonl"
DONE_FILE="$BULK_DIR/pmc_bundles_done.txt"
LOG="$BULK_DIR/pmc_extract.log"

# PMC subsets to process
SUBSETS=("oa_comm" "oa_noncomm" "oa_other")

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-gb)  MAX_OUTPUT_GB="$2"; shift 2 ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$BULK_DIR"
touch "$DONE_FILE"

echo "PMC full-text extraction" | tee -a "$LOG"
echo "  Max output size: ${MAX_OUTPUT_GB} GB" | tee -a "$LOG"
echo "  Output: $OUTPUT" | tee -a "$LOG"
echo "  Subsets: ${SUBSETS[*]}" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

STOPPED=0
for SUBSET in "${SUBSETS[@]}"; do
    if [ "$STOPPED" -eq 1 ]; then
        break
    fi

    PMC_BASE="https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/${SUBSET}/xml"
    BUNDLE_LIST=$(curl -sL "$PMC_BASE/" \
        | grep -oP "${SUBSET}_xml\.[^\"]+baseline[^\"]+\.tar\.gz" | sort -u)

    TOTAL_BUNDLES=$(echo "$BUNDLE_LIST" | wc -l)
    echo "" | tee -a "$LOG"
    echo "=== Subset: $SUBSET ($TOTAL_BUNDLES bundles) ===" | tee -a "$LOG"

    BUNDLE_NUM=0
    for BUNDLE in $BUNDLE_LIST; do
        BUNDLE_NUM=$((BUNDLE_NUM + 1))

        # Skip if already processed (done-file entries are bundle filenames)
        if grep -qF "$BUNDLE" "$DONE_FILE" 2>/dev/null; then
            continue
        fi

        # Check output size limit
        if [ -f "$OUTPUT" ]; then
            OUTPUT_BYTES=$(stat --format=%s "$OUTPUT" 2>/dev/null || echo 0)
            OUTPUT_GB=$(python3 -c "print(f'{$OUTPUT_BYTES / 1024**3:.2f}')")
            if python3 -c "exit(0 if $OUTPUT_BYTES >= $MAX_OUTPUT_GB * 1024**3 else 1)" 2>/dev/null; then
                echo "  Output size limit reached: ${OUTPUT_GB} GB >= ${MAX_OUTPUT_GB} GB. Stopping." | tee -a "$LOG"
                STOPPED=1
                break
            fi
        fi

        URL="$PMC_BASE/$BUNDLE"
        TEMP_TAR="$BULK_DIR/_temp_bundle.tar.gz"

        echo "  [$SUBSET $BUNDLE_NUM/$TOTAL_BUNDLES] Downloading $BUNDLE..." | tee -a "$LOG"

        # Download bundle
        if ! wget -q --timeout=120 --tries=3 -O "$TEMP_TAR" "$URL" 2>/dev/null; then
            echo "    FAILED to download, skipping" | tee -a "$LOG"
            rm -f "$TEMP_TAR"
            continue
        fi

        BUNDLE_SIZE=$(du -h "$TEMP_TAR" | cut -f1)
        echo "    Downloaded: $BUNDLE_SIZE" | tee -a "$LOG"

        # Extract text from matching articles and append to JSONL
        EXTRACTED=$(python3 -c "
import tarfile, gzip, sys, json
from lxml import etree

# Load PMIDs we care about (compound-linked)
pmid_file = '$BULK_DIR/CID-PMID.gz'
target_pmids = set()
with gzip.open(pmid_file, 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            target_pmids.add(parts[1])

count = 0
with tarfile.open('$TEMP_TAR', 'r:gz') as tar:
    with open('$OUTPUT', 'a', encoding='utf-8') as out:
        for member in tar:
            if not member.name.endswith('.xml') and not member.name.endswith('.nxml'):
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                xml_bytes = f.read()
                root = etree.fromstring(xml_bytes)

                # Get PMID
                pmid_elem = root.find('.//article-id[@pub-id-type=\"pmid\"]')
                if pmid_elem is None or pmid_elem.text is None:
                    continue
                pmid = pmid_elem.text.strip()

                # Skip if not linked to a compound
                if pmid not in target_pmids:
                    continue

                # Extract text
                title_el = root.find('.//article-title')
                title = ''.join(title_el.itertext()).strip() if title_el is not None else ''
                if not title:
                    continue

                abstract_el = root.find('.//abstract')
                abstract = ''.join(abstract_el.itertext()).strip() if abstract_el is not None else ''

                body_el = root.find('.//body')
                full_text = ''.join(body_el.itertext()).strip() if body_el is not None else ''

                sections = {}
                for sec in root.findall('.//body/sec'):
                    sec_title_el = sec.find('title')
                    if sec_title_el is not None:
                        sec_title = ''.join(sec_title_el.itertext()).strip()
                        sec_text = ''.join(sec.itertext()).strip()
                        if sec_title and sec_text:
                            sections[sec_title] = sec_text

                # PMC ID
                pmcid_elem = root.find('.//article-id[@pub-id-type=\"pmc\"]')
                pmcid = f'PMC{pmcid_elem.text.strip()}' if pmcid_elem is not None and pmcid_elem.text else ''

                # DOI
                doi_elem = root.find('.//article-id[@pub-id-type=\"doi\"]')
                doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ''

                # Journal
                journal_el = root.find('.//journal-title')
                journal = ''.join(journal_el.itertext()).strip() if journal_el is not None else ''

                doc = {
                    'pmid': pmid,
                    'pmcid': pmcid,
                    'doi': doi,
                    'title': title,
                    'abstract': abstract,
                    'full_text': full_text,
                    'sections': sections,
                    'journal': journal,
                    'pmc_subset': '$SUBSET',
                }

                out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                count += 1

            except Exception:
                continue

print(count)
" 2>>"$LOG")

        echo "    Extracted: $EXTRACTED articles" | tee -a "$LOG"

        # Clean up bundle
        rm -f "$TEMP_TAR"

        # Mark as done
        echo "$BUNDLE" >> "$DONE_FILE"

        # Report output size
        if [ -f "$OUTPUT" ]; then
            OUT_SIZE=$(du -h "$OUTPUT" | cut -f1)
            OUT_LINES=$(wc -l < "$OUTPUT")
            echo "    Output so far: $OUT_LINES articles, $OUT_SIZE" | tee -a "$LOG"
        fi
    done
done

echo "---" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
if [ -f "$OUTPUT" ]; then
    echo "Total articles: $(wc -l < "$OUTPUT")" | tee -a "$LOG"
    echo "Output size: $(du -h "$OUTPUT" | cut -f1)" | tee -a "$LOG"
fi
