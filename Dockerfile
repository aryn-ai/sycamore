FROM nikolaik/python-nodejs:python3.11-nodejs20

COPY ui /home/pn/js-ui
COPY openai-proxy /home/pn/py-proxy
COPY run-ui.sh /home/pn

WORKDIR /home/pn/js-ui
RUN npm install

WORKDIR /home/pn/py-proxy
RUN poetry install

WORKDIR /home/pn
CMD ./run-ui.sh
