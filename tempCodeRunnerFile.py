import subprocess
import os

def run_command(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr)
    else:
        print(result.stdout)

os.chdir(r'C:\Users\eason\Documents\All\projects\3DPrints\Robot_Muchi\code\Data_Processing')

commands = [
    "git init",
    "git add .",
    'git commit -m "first commit"',
    "git pull origin master --allow-unrelated-histories",  
    "git push origin master"
]


for command in commands:
    run_command(command)