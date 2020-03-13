import waws
import os

s3 = waws.BucketManager()

import os

path = os.getcwd()

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        #if ".iml" not in file and ".xml" not in file:
        # Upload files
        s3.upload_file(
            file_name=file,
            local_path=r,
            remote_path="~/retrograph"
        )


