FROM jupyter/scipy-notebook

RUN pip3 install -r requirements.txt
WORKDIR /app
ADD . /app
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
