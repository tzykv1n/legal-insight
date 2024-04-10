import subprocess

result = subprocess.run(['winget', 'install', '--id' , '9NRWMJP3717K'], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("Error:", result.stderr) 