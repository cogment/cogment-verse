# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

from urllib.parse import urlparse

from ..services_directory import ServiceType


def build_prometheus_http_sd(service_directory):
    prometheus_http_sd = []
    prometheus_service_names = service_directory.get_service_names(ServiceType.PROMETHEUS)

    for prometheus_service_name in prometheus_service_names:
        prometheus_service_url = service_directory.get(ServiceType.PROMETHEUS, prometheus_service_name)

        prometheus_service_url_components = urlparse(prometheus_service_url)

        if prometheus_service_url_components.scheme != "http":
            raise RuntimeError(f"Unsupported scheme for [{prometheus_service_url}], expected 'http'")

        prometheus_http_sd.append(
            {"targets": [prometheus_service_url_components.netloc], "labels": {"service": prometheus_service_name}}
        )

    return prometheus_http_sd
