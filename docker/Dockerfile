
FROM huggingface/transformers-pytorch-gpu
LABEL maintainer="Patrick Blackman Sphaier"

RUN python3 -m pip install --no-cache-dir jupyter
RUN python3 -m pip install matplotlib pandas torchtext sklearn seaborn
EXPOSE 9203

VOLUME /data
WORKDIR /data

# notebook password: fastai
CMD /bin/bash -c "jupyter notebook --ip='0.0.0.0' --port=9203 --allow-root --no-browser --NotebookApp.password='sha1:a60ff295d0b9:506732d050d4f50bfac9b6d6f37ea6b86348f4ed'"

