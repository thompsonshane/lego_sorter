#   imports:
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

#_______________________________________________________________________________________________

# This script will:
# 1.    Download this zip file
# 2.    Unzip it
# 3.    Extract the individual models
# 4.    Neatly organise the model data for ease of use

# 1,2,3

zipurl = "http://www.ldraw.org/library/updates/complete.zip"

with urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('./models')

# 4. the zip file has been extracted to the project and now the parts must be organised individually