# Project

This project will be an extended work on C:\Users\m.rahman\PythonProjects\tillage\resource\tong.pdf.

On that paper, the authors claimed : "Preprocessing choice for MIR-based SOC estimation should be tailored to the target domain/region, rather than treated as universally optimal."

My criticism is that they used PLSR in their experiment and that is why this dependency. If they had used some smarted algorithm like DL, it might have been different.

For that purpose, you need to understand their experiment, dataset, filter, split - everything 100% accurately. I will reproduce their experiment by replacing RLSR with DL, and a few more potential replacement.

I am telling again. Your highest priority will be understanding their experimental parameters. If we do everything else well and publish the paper with fantastic result but later it was found that there was a minor mismatch in experiment details (which was not mentioned/noticed), I will be in trouble. You must not miss any single detail about that.

# Tools
Use C:\Users\m.rahman\vens\tillage venv ("C:\Users\m.rahman\vens\tillage\Scripts\python.exe") as the project venv. In a way so that when I open a terminal that venc is automatically activated. Also, when I right click a file and click run in python, this venv is used. You might need to create .vscode and write necessary config files for it. 