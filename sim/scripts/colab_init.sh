#!/bin/bash
# Source this file at the start of a Colab session:
#   source colab_init.sh
# Or upload it to Colab and run:
#   source /content/colab_init.sh

git clone https://github.com/idanhochman/capim_.git /content/capim_

pip install bitsandbytes
pip install "transformers<5.0.0"

huggingface-cli login
