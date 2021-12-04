FROM jupyter/scipy-notebook

RUN pip install scikit-learn
RUN pip3 install qiskit
RUN pip3 install pylatexenc
RUN pip3 install plotly
RUN pip3 install mlxtend
RUN pip3 install missingno
RUN pip3 install matplotlib 
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install pandas
WORKDIR /app
ADD . /app
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
