#!/usr/bin/env bash



iconv -c -f "utf-8" -t "utf-8" ./CBOW_Vector > ./utf8_CBOW_Vector

iconv -c -f "utf-8" -t "utf-8" ./SKIP_GRAM_Vector > ./utf8_SKIP_GRAM_Vector


