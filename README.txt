How to install and run the project
-----------------------------------

1. Open the VSCode terminal  
   - Click on the "Run" icon to open the terminal.

2. Create a virtual environment by typing:  
   python -m venv .venv

3. Activate the virtual environment:  
   - For macOS/Linux:  
     source .venv/bin/activate  
   - For Windows:  
     .venv\Scripts\activate  

   If the virtual environment is successfully activated, you will see (.venv) at the beginning of the terminal line.

   NOTE: If you encounter an error during activation (step 3) such as UnauthorizedAccess, type the following command in the VSCode terminal:  
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  
   Then retry the activation command from step 3.

4. Install all required packages for the script by typing:  
   python -m pip install -r requirements.txt

5. Select the virtual environment interpreter instead of the default Python interpreter.  
   To do so, click on the python version you have (bottom right of your screen) and select the virtual environment instead.

6. Run the file `main.py` by pressing "Run".  
   The process should now start.

Troubleshooting:
----------------
- If you see the error "No module named dcm2niix", this may occur if you are using the laboratory computers.  
  Package installation does not always place executables in the PATH. This is an issue for `dcm2niix` because it is executed from a subprocess.

  Possible solutions:
  1. Try replacing line 221 of `class_segmentation.py` (as shown in Figure 3.4) with simply:  
     "python"  
  2. If that does not work, add `dcm2niix.exe` to your PATH.