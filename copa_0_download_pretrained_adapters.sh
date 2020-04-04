#! /bin/bash
waws --downloadS3 -f 1.0_1.0_5_30_full_assertions_nl.zip -b wluper-retrograph
unzip 1.0_1.0_5_30_full_assertions_nl.zip
mv 1.0_1.0_5_30_full_assertions_nl.zip models
mv 1.0_1.0_5_30_full_assertions_nl models
