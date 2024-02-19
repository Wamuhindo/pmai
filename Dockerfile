FROM python:3.10.0

RUN apt-get update && apt-get install libgl1 -y 

RUN mkdir -p /home/PMAIEDGE
WORKDIR /home/PMAIEDGE

ADD . /home/PMAIEDGE/
RUN pip3 install -r requirements_GPU.txt

RUN pip install --no-use-pep517 -e ./mushroom_rl/.

CMD bash

#ENTRYPOINT ["python", "./main_checkpoint.py"]
#CMD["-c", "--start_from_checkpoint", "-D"]
