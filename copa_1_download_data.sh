#! /bin/bash
waws --downloadS3 -f copa_en.zip -b wluper-retrograph
mkdir data/COPA
unzip copa_en.zip
mv test_gold.jsonl data/COPA
mv train.en.jsonl data/COPA
mv val.en.jsonl data/COPA
mv copa_en.zip data
