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

## Creating a Cloud Instance
Estimated time to complete: 5 mins

*You can also follow our video tutorial to set up a cloud instance on Lambda* 👉️ [YouTube Video](https://www.youtube.com/watch?v=Ndm9CROuk5g&list=PLo2EIpI_JMQtncHQHdHq2cinRVk_VZdGW)

1. Click the link: https://cloud.lambdalabs.com/instances
2. You'll be asked to sign in to your Lambda account (if you haven't done so already).
3. Once on the GPU instance page, click the purple button "Launch instance" in the top right.
4. Verify a payment method if you haven't done so already.
5. Launching an instance:
   1. In "Instance type", select the instance type "1x A100 (40 GB SXM4)"
   2. In "Select region", select the region with availability closest to you.
   3. In "Select filesystem", select "Don't attach a filesystem".
6. You will be asked to provide your public SSH key. This will allow you to SSH into the GPU device from your local machine.
   1. If you’ve not already created an SSH key pair, you can do so with the following command from your local device: 
      ```bash
      ssh-keygen
      ```
   2. You can find your public SSH key using the command: 
      ```bash
      cat ~/.ssh/id_rsa.pub
      ```
      (Windows: `type C:UsersUSERNAME.sshid_rsa.pub` where `USERNAME` is the name of your user)
   4. Copy and paste the output of this command into the first text box
   5. Give your SSH key a memorable name (e.g. `sanchits-mbp`)
   6. Click "Add SSH Key"
7. Select the SSH key from the drop-down menu and click "Launch instance"
8. Read the terms of use and agree
9. We can now see on the "GPU instances" page that our device is booting up!
10. Once the device status changes to "✅ Running", click on the SSH login ("ssh ubuntu@..."). This will copy the SSH login to your clipboard.
11. Now open a new command line window, paste the SSH login, and hit Enter.
12. If asked "Are you sure you want to continue connecting?", type "yes" and press Enter.
13. Great! You're now SSH'd into your A100 device! We're now ready to set up our Python environment!

You can see your total GPU usage from the Lambda cloud interface: https://cloud.lambdalabs.com/usage

Here, you can see the total charges that you have incurred since the start of the event. We advise that you check your 
total on a daily basis to make sure that it remains below the credit allocation of $110. This ensures that you are 
not inadvertently charged for GPU hours.

If you are unable to SSH into your Lambda GPU in step 11, there is a workaround that you can try. On the [GPU instances page](https://cloud.lambdalabs.com/instances), 
under the column "Cloud IDE", click the button "Launch". This will launch a Jupyter Lab on your GPU which will be displayed in your browser. In the 
top left-hand corner, click "File" -> "New" -> "Terminal". This will open up a new terminal window. You can use this 
terminal window to set up your Python environment in the next section [Set Up an Environment](#set-up-an-environment).

## Deleting a Cloud Instance

100 1x A100 hours should provide you with enough time for 5-10 fine-tuning runs (depending on how long you train for 
and which size models). To maximise the GPU time you have for training, we advise that you shut down GPUs over prolonged 
periods of time when they are not in use. Leaving a GPU running accidentally over the weekend will incur 48 hours of 
wasted GPU hours. That's nearly half of your compute allocation! So be smart and shut down your GPU when you're not training.

Creating an instance and setting it up for the first time may take up to 20 minutes. Subsequently, this process will 
be much faster as you gain familiarity with the steps, so you shouldn't worry about having to delete a GPU and spinning one 
up the next time you need one. You can expect to spin-up and delete 2-3 GPUs over the course of the fine-tuning event.


We'll quickly run through the steps for deleting a Lambda GPU. You can come back to these steps after you've 
performed your first training run and you want to shut down the GPU:

1. Go to the instances page: https://cloud.lambdalabs.com/instances
2. Click the checkbox on the left next to the GPU device you want to delete
3. Click the button "Terminate" in the top right-hand side of your screen (under the purple button "Launch instance")
4. Type "erase data on instance" in the text box and press "ok"

Your GPU device is now deleted and will stop consuming GPU credits.

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

Next, I'll guide `LD_LIBRARY_PATH` towards the path of the CUDA libraries. This step tells the shared run-time library loader where to find the CUDA directory:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libcudart.so
```

Now source the `.bashrc` for the changes to take effect

```bash
source ~/.bashrc
```

I then try running the command again 🤞🏽

```bash
cd ~
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

If you're eager to experiment with the fine-tuned PEFT weights, initialize Gradio using this command from `peft-training` root directory:

```bash
python interface.py --model-name {model} --adapters-name {adapter}
```

Where `model` is the name of the model to load, and `adapter` is the name of the adapter to load. The defaults are `decapoda-research/llama-7b-hf` and `timdettmers/guanaco-7b` , respectively.

## 📚 Useful Resources

The following libraries, which are referenced throughout this document, can be useful for further exploration 🐱‍👤:

https://github.com/TimDettmers/bitsandbytes

https://github.com/huggingface/peft
