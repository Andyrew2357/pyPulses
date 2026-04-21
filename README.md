# pyPulses
Instrument control code for pulsed electronic measurements, intended for use in an IPython notebook.

## Quick Installation Instructions
1. Clone the repository: <code>git clone https://github.com/Andyrew2357/pyPulses.git</code>
2. Create a virtual environment in your intended operating directory and activate that environment.
3. Navigate to the cloned repository and run: <code>python -m pip install -e .</code>. The <code>-e</code> flag will make this an editable install. Therefore any changes to the cloned repository will be reflected upon importing.
4. To use the package, create a jupyter notebook and import from <code>pyPulses</code>, <code>pyPulses.devices</code>, <code>pyPulses.utils</code>, etc.
5. If you are working in Visual Studio Code, it helps to edit your <code>settings.json</code> to include <code>"python.analysis.extraPaths": ["path/to/cloned/repository"]</code> and <code>"python.languageServer": "Pylance"</code>. This will ensure proper code highlighting and IntelliSense.
