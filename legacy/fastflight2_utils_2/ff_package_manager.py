import os
import zipfile
import urllib.request
import subprocess
import tempfile
import shutil
from pathlib import Path

class FastFlightPackageManager:
    """
    Manages a portable 32-bit Python environment for FastFlight DLL 
    communication. Downloads and sets up everything needed without user 
    intervention.
    """
    
    def __init__(self, install_dir: str = None):
        if install_dir is None:
            install_dir = os.path.join(os.path.expanduser("~"), ".fastflight")
        
        self.install_dir = Path(install_dir)
        self.python32_dir = self.install_dir / "python32"
        self.python32_exe = self.python32_dir / "python.exe"
        self.scripts_dir = self.install_dir / "scripts"
        
        # Embedded Python download URL (32-bit)
        self.python_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-win32.zip"
        self.get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        
    def is_installed(self) -> bool:
        """Check if the 32-bit environment is already set up"""
        return (self.python32_exe.exists() and \
                (self.scripts_dir / "ff_bridge_server.py").exists())
    
    def install(self):
        """Download and set up the complete 32-bit Python environment"""
        if self.is_installed():
            return str(self.python32_exe)
        
        # Create directories
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract embedded Python
        self._download_python()
        
        # Install pip and required packages
        self._setup_pip()
        self._install_packages()
        
        # Copy FastFlight scripts
        self._copy_scripts()

        return str(self.python32_exe)
    
    def _download_python(self):
        """Download and extract embedded Python 32-bit"""
        
        with tempfile.NamedTemporaryFile(suffix='.zip', 
                                         delete=False) as tmp_file:
            urllib.request.urlretrieve(self.python_url, tmp_file.name)
            
            # Extract to python32 directory
            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                zip_ref.extractall(self.python32_dir)
        
        os.unlink(tmp_file.name)
        
        # Enable pip by modifying pth file
        pth_file = self.python32_dir / "python311._pth"
        if pth_file.exists():
            content = pth_file.read_text()
            if "#import site" in content:
                content = content.replace("#import site", "import site")
                pth_file.write_text(content)
    
    def _setup_pip(self):
        """Install pip in the embedded Python"""
        
        with tempfile.NamedTemporaryFile(suffix='.py', 
                                         delete=False) as tmp_file:
            urllib.request.urlretrieve(self.get_pip_url, tmp_file.name)
            
            # Run get-pip.py with the embedded Python
            subprocess.run([str(self.python32_exe), tmp_file.name], 
                         cwd=str(self.python32_dir), check=True)
        
        os.unlink(tmp_file.name)
    
    def _install_packages(self):
        """Install required packages (comtypes)"""
        
        subprocess.run([
            str(self.python32_exe), "-m", "pip", "install", "comtypes", "numpy"
        ], check=True)
    
    def _copy_scripts(self):
        """Copy FastFlight Python scripts to the scripts directory"""
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        
        # Copy ff_bridge_server.py
        src_bridge = current_dir / "ff_bridge_server.py"
        if src_bridge.exists():
            shutil.copy2(src_bridge, self.scripts_dir / "ff_bridge_server.py")

    def get_python32_path(self) -> str:
        """Get the path to the 32-bit Python executable"""
        if not self.is_installed():
            print("Installing 32-bit Python executable...")
            return self.install()
        return str(self.python32_exe)

    def get_bridge_script_path(self) -> str:
        """Get the path to the bridge server script"""
        return str(self.scripts_dir / "ff_bridge_server.py")
    
    def uninstall(self):
        """Remove the entire FastFlight installation"""
        if self.install_dir.exists():
            shutil.rmtree(self.install_dir)
