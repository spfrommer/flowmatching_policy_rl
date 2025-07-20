# curl -fsSL https://pyenv.run | bash

# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
# echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

# sudo apt update; sudo apt install -y build-essential libssl-dev zlib1g-dev \
# libbz2-dev libreadline-dev libsqlite3-dev curl git \
# libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# source ~/.bashrc
# pyenv install 3.12.10

mkvirtualenv flowmatchingrl -p ~/.pyenv/versions/3.12.10/bin/python3.12
pip install -e .