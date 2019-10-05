#! /bin/bash

pip install jupyter~=1.0.0 jupyter-http-over-ws~=0.0.6

jupyter serverextension enable --py jupyter_http_over_ws
jupyter nbextension enable --py --sys-prefix widgetsnbextension
