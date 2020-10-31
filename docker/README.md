docker build --tag nlp-workbench .
nvidia-docker run -dv /home/pblackman/:/data/ -p 9203:9203 nlp-workbench
