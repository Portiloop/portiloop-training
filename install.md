This is a step-by-step guide to install the project, it is very detailed to be accessible to everyone.

# Create an env

If you know how to create an env do it the way you want. This is just a guide for people who don't.

#### If not installed yet install virtualenv

> pip install virtualenv

#### Create an env

Standart names: .env, venv, env, .venv...

>  python<version> -m venv <virtual-environment-name>

#### Activate env

>  source <virtual-environment-name>/bin/activate

# Setup project

Run this command in the project folder

> pip install .

# Weight and Biases credentials

Open the activate file in your env and add your credentials by adding the following lines:
(you can find your credentials in your profile)

> export WANDB_API_KEY=<your-api-key> \
> export WANDB_PROJECT=<your-project-name> \
> export WANDB_USERNAME=<your-username>

Restart your env.

> deactivate \
> source <virtual-environment-name>/bin/activate

Make sure Weight and Biases is in online mode

> wandb online

# How to run

You need to have internet connection to use Weight and Biases and you device needs to be able to use CUDA.

The model will train and adapt with every file in the portiloopml/dataset directory.

To run with sbatch:

> sbatch train_job.sh

You can edit the train_job file to your liking.
All parameters can be found in the portiloopml/portiloop_python/ANN/portiloop_detector_training.py main function docstring.

If you wish to run with salloc you can use the parameters of the sbatch in the file train_job.sh and then use the command:

> bash train_job.sh
