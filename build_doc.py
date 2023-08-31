"""
Build the documentation using pdoc (not pdoc3)
"""
import shutil
import glob
import os

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")
os.mkdir("./docs/hqm/")

# Build documentation
os.system("pdoc ./hqm -o ./docs --docformat numpy --logo logo.png -t ./docs_assets/") 
shutil.copy2("./docs_assets/logo.png", "./docs/logo.png")
shutil.copy2("./docs_assets/logo.png", "./docs/hqm/logo.png")
#shutil.copy2("./docs_assets/favicon.ico", "./docs/favicon.ico")

for p in glob.glob("./docs/hqm/*/"):
    shutil.copy2("./docs_assets/logo.png", p+"/logo.png")