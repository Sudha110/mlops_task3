FROM centos:7
RUN yum install python36 -y
RUN yum install epel-release -y
RUN yum install python36-devel -y
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install tensorflow==1.5
RUN pip3 install keras==2.1.5
RUN pip3 install keras
