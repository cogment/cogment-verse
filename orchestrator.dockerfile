FROM cogment/orchestrator:v2.0.0

RUN apt-get update && apt-get -y install gettext-base

ADD cogment.yaml ./
ADD *.proto ./

ENV COGMENT_LIFECYCLE_PORT=9000
ENV COGMENT_ACTOR_PORT=9000
#ENV PROMETHEUS_PORT=8000

# Didn't manage to setup envsubst in the ENTRYPOINT
ENTRYPOINT []
CMD envsubst < /app/cogment.yaml > /app/cogment.yaml.out && orchestrator --config=/app/cogment.yaml.out --params=/app/cogment.yaml.out
