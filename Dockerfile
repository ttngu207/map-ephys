FROM datajoint/pydev:python3.6

RUN apt update && apt -y install mysql-client-5.7 netcat

#RUN pip install globus_sdk

RUN mkdir /dj_map
WORKDIR /dj_map

RUN git clone https://github.com/mesoscale-activity-map/map-ephys /dj_map

RUN pip install .