import subprocess
import os

def run_command(command):
    """Run a shell command with the current environment variables."""
    # Copy the current environment variables
    env = os.environ.copy()

    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, env=env)
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\nExit code: {e.returncode}\nOutput: {e.output.decode()}")

def main(): 
    # Change directory to input
    os.chdir("input")

    # Clone the Faiss repository
    if not os.path.exists("faiss"):
        run_command("proxychains git clone https://github.com/facebookresearch/faiss.git")
    
    os.chdir("faiss")
    
    # Create a build directory
    if not os.path.exists("build"):
        os.mkdir("build")
    os.chdir("build")
    
    # Configure with CMake
    cuda_architectures = "Auto"  # Adjust based on your GPU. You can specify explicitly, e.g., "70" for Tesla V100
    run_command(f"cmake -B . -DFAISS_ENABLE_GPU=ON -DCUDA_ARCHITECTURES={cuda_architectures} -DBUILD_TESTING=OFF -DFAISS_ENABLE_PYTHON=ON ..")
    
    # Build Faiss
    run_command("make -j $(nproc)")
    
    # Install the Faiss Python package
    os.chdir("../python")
    run_command("python setup.py install")
    
    print("Faiss build and installation completed.")

if __name__ == "__main__":
    main()
