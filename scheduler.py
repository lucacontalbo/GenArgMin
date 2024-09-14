import subprocess
import time

def run_command():
    try:
        command = ['python3', 'main.py', '--dataset', 'student_essay', '--seed', '0', '--use_graph', '--batch_size', '8']
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    while True:
        success = run_command()
        if success:
            print("Command executed successfully. Exiting program.")
            break
        else:
            print("Command failed. Waiting 30 minutes before next attempt.")
            time.sleep(10 * 60)

if __name__ == "__main__":
    main()
