import subprocess
import os

docs_dir = "docs-site"  # Change this to the directory where mkdocs.yml sits
os.chdir(docs_dir)
subprocess.run(["mkdocs", "serve"])
