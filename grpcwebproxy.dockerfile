FROM golang:1.17 as dev

WORKDIR /go

ENV GOPATH=/go

RUN go get github.com/improbable-eng/grpc-web/go/grpcwebproxy@v0.14.1

CMD grpcwebproxy --server_http_max_read_timeout 8760h --server_http_max_write_timeout 8760h --backend_addr=$COGMENT_VERSE_ORCHESTRATOR_ENDPOINT --run_tls_server=false --allow_all_origins --use_websockets --server_http_debug_port=$COGMENT_VERSE_GRPCWEBPROXY_PORT
