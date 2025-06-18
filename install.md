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