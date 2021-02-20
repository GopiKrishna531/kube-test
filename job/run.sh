# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate /storage/home/gopikrishna/.conda/envs/g_py
# Change to the directory in which your code is present
cd /storage/home/gopikrishna/kube-test/job
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.

python -u generate_images_per_class.py &> /storage/home/gopikrishna/out
