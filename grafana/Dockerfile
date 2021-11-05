FROM grafana/grafana

COPY ./grafana.ini /etc/grafana

COPY ./provisioning /provisioning
ENV GF_PATHS_PROVISIONING /provisioning

COPY ./dashboards /dashboards
ENV COGMENT_VERSE_GRAFANA_DASHBOARDS_DIR /dashboards

VOLUME /data
ENV GF_PATHS_DATA /data
