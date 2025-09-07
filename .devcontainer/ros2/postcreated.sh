#!/bin/bash

# 호스트의 IP 주소 가져오기
HOST_IP=$(ip route show default | awk '/default/ {print $3}')

# 호스트의 IP 주소를 호스트 이름에 추가
echo "$HOST_IP host.docker.internal" >> /etc/hosts
