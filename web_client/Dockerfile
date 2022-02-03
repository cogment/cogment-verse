# "dev" is the developpment image
#FROM node:14 as dev
FROM node:14 as dev

# install git
RUN set -x \
    && . /etc/os-release \
    apt-get update \
    && apt-get -yq install bash git \
    && apt-get -yq clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # show installed application versions
    && git --version && bash --version && ssh -V && npm -v && node -v && yarn -v

WORKDIR /cogment-verse/web_client

# Install dependencies
COPY package.json package-lock.json cogment.yaml data.proto ./
RUN npm install

# Use polling to catch file changes from inside the container
# cf. https://stackoverflow.com/questions/44643045/running-development-server-with-create-react-app-inside-of-a-docker-container
ENV CHOKIDAR_USEPOLLING=true

COPY . .

# start the web client in development mode
CMD ["sh", "-c", "npx cogment-js-sdk-generate cogment.yaml && npm start"]

# "prod" is the production image
FROM dev as prod

RUN npx cogment-js-sdk-generate cogment.yaml
RUN npm run build

# start image in production mode
CMD npx serve -s -l ${PORT} build
