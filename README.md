# Parameter-Efficient Fine-Tuning or “PEFT”

Created time: June 2, 2023 7:34 PM <br>
Last edited time: June 4, 2023 6:40 PM <br>
Owner: Jason Wheeler <br>
Tags: Compute, LLMs, Memory, Training

## 🚀 Setting the Stage

To kick things off, we need to clone the peft-training repository:

```bash
git clone https://github.com/lightshifted/peft-training
```

I execute this command to install project dependencies:

```markdown
cd peft-training
pip install -r requirements.txt
```

## 🎛 Configuring BitsandBytes in LambdaLabs SSH

When first attempting to run BitsandBytes in a new LambdaLabs environment, you will likely encounter this issue:

[https://github.com/TimDettmers/bitsandbytes/issues/156](https://github.com/TimDettmers/bitsandbytes/issues/156)

As a solution, I copy `libbitsandbytes_cuda117.so` to `libbitsandbytes_cpu.so` by executing this command:

```bash
cd /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/
cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so
```

To verify whether CUDA libraries exist in the ssh, I execute this command from the same directory

```bash
find / -name libcudart.so 2>/dev/null
```

I've discovered CUDA libraries at `/usr/lib/x86_64-linux-gnu/libcudart.so`! 🔥

Next, I'll guide LD_LIBRARY_PATH towards the path of the CUDA libraries. This step tells the shared run-time library loader where to find the CUDA directory:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libcudart.so
```

Now source the `.bashrc` for the changes to take effect

```bash
source ~/.bashrc
```

I then try running the command again 🤞🏽

```bash
cd ~/peft-training
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

## Engaging in Conversations with the Fine-Tuned Model

To ensure smooth compatability with asynchronous operations, I upgrade `fsspec`:

```bash
pip install --upgrade fsspec
```

If you're eager to experiment with the fine-tuned PEFT weights, initialize Gradio using this command:

```bash
~/peft-training$ python interface --model-name {model} --adapters-name {adapter}
```

Where `model` is the name of the model to load, and `adapter` is the name of the adapter to load. The defaults are `decapoda-research/llama-7b-hf` and `timdettmers/guanaco-7b` , respectively.

## 📚 Useful Resources

The following libraries, which are referenced throughout this document, can be useful for further exploration 🐱‍👤:

https://github.com/TimDettmers/bitsandbytes

https://github.com/huggingface/peft