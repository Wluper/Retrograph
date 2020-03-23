#! /bin/bash
waws --downloadS3 -f omcs_pretraining_free_wo_nsp_adapter.zip -b wluper-retrograph
tar -xf omcs_pretraining_free_wo_nsp_adapter.zip
mv omcs_pretraining_free_wo_nsp_adapter.zip models
mv omcs_pretraining_free_wo_nsp_adapter models
