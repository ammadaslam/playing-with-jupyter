FROM jupyter/scipy-notebook

RUN pip install scikit-learn
RUN pip3 install qiskit
RUN pip3 install pylatexenc
RUN pip3 install plotly
RUN pip3 install mlxtend
RUN pip3 install missingno
WORKDIR /app
ADD . /app



