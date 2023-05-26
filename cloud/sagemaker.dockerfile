# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y swig python3-opencv python3-pip nodejs npm

RUN mkdir -p /opt/ml/code/cloud/
COPY requirements.txt /opt/ml/code/
COPY cloud/sagemaker_utils.py /opt/ml/code/cloud/
COPY cloud/sagemaker_main.py /opt/ml/code/
RUN chmod +x /opt/ml/code

WORKDIR /opt/ml/code
RUN pip install --no-cache-dir -r requirements.txt --timeout 5000
RUN pip install SuperSuit==3.7.0
RUN chmod +x sagemaker_main.py

# Activate license for atari game
RUN AutoROM --accept-license

ENTRYPOINT ["python", "sagemaker_main.py"]

