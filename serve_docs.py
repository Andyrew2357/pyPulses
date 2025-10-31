import subprocess
import os

docs_dir = "docs-site"
os.chdir(docs_dir)
subprocess.run(["mkdocs", "serve"])
