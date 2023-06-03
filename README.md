# Parameter-Efficient Fine-Tuning or “PEFT”

Created time: June 2, 2023 7:34 PM
Last edited time: June 2, 2023 8:40 PM
Owner: Jason Wheeler
Tags: Compute, LLMs, Memory, Training

## Setting up BitsandBytes in LamdaLabs SSH

I execute this command from the root of the `peft-training` directory:

```bash
cd /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/
cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so
```

To verify whether CUDA libraries exist in the ssh, I execute this command from the same directory

```bash
find / -name libcudart.so 2>/dev/null
```

Found at `/usr/lib/x86_64-linux-gnu/libcudart.so`! 🔥

I next set `LD_LIBRARY_PATH` to the CUDA libraries path; this gives the shared run-time library loader the CUDA directory to search for:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libcudart.so
```

Now source the `.bashrc` for the changes to take effect

```bash
source ~/.bashrc
```

I then try running the command again 🤞🏽

```bash
python -m bitsandbytes
```

And voila!

```bash
ubuntu@104-171-203-54:~/peft-training$ python -m bitsandbytes

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: :/usr/lib/x86_64-linux-gnu/libcudart.so did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/local/cuda/lib64')}
  warn(msg)
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!
  warn(msg)
CUDA SETUP: Highest compute capability among GPUs detected: 7.5
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++

++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['7.5']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable

WARNING: Please be sure to sanitize sensible info from any such env vars!

SUCCESS!
Installation was successful!
```