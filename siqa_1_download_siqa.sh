#! /bin/bash
waws --downloadS3 -f socialIQa_v1.4.zip -b wluper-retrograph
mkdir data/SIQA
unzip socialIQa_v1.4.zip
mv socialIQa_v1.4_dev.jsonl data/SIQA
mv socialIQa_v1.4_trn.jsonl data/SIQA
mv socialIQa_v1.4_tst.jsonl data/SIQA
mv socialIQa_v1.4.zip data
