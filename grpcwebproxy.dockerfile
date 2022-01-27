FROM golang:1.17 as dev

WORKDIR /go

ENV GOPATH=/go

RUN go get github.com/improbable-eng/grpc-web/go/grpcwebproxy@v0.14.1

CMD grpcwebproxy --backend_addr=$COGMENT_VERSE_ORCHESTRATOR_ENDPOINT --run_tls_server=false --allow_all_origins --use_websockets --server_http_debug_port=$COGMENT_VERSE_GRPCWEBPROXY_PORT
