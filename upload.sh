#!/bin/bash
set -e
cd /hy-tmp
file="results-$(date "+%Y%m%d-%H%M%S").zip"
zip -q -r "${file}" results
oss cp "${file}" oss://backup/
rm -f "${file}"
shutdown
