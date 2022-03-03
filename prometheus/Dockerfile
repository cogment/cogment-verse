FROM prom/prometheus
COPY --from=hairyhenderson/gomplate:stable /gomplate /bin/gomplate

ADD prometheus.yml.tmpl /etc/prometheus/

VOLUME /prometheus

ENTRYPOINT []
CMD \
  /bin/gomplate -f /etc/prometheus/prometheus.yml.tmpl -o /etc/prometheus/prometheus.yml && \
  /bin/prometheus \
  --config.file /etc/prometheus/prometheus.yml \
  --log.level error \
  --storage.tsdb.path=/prometheus \
  --web.listen-address="0.0.0.0:${COGMENT_VERSE_PROMETHEUS_PORT}" \
  --web.console.libraries=/usr/share/prometheus/console_libraries \
  --web.console.templates=/usr/share/prometheus/consoles
