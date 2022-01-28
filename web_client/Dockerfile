# "dev" is the developpment image
#FROM node:14 as dev
FROM node:14 as dev

# install git
RUN set -x \
    && . /etc/os-release \
    && case "$ID" in \
    alpine) \
    apk add --no-cache bash git openssh \
    ;; \
    debian) \
    apt-get update \
    && apt-get -yq install bash git openssh-server \
    && apt-get -yq clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    ;; \
    esac \
    # install yarn, if needed (only applies to older versions, like 6 or 7)
    && yarn bin || ( npm install --global yarn && npm cache clean ) \
    # show installed application versions
    && git --version && bash --version && ssh -V && npm -v && node -v && yarn -v

WORKDIR /cogment-verse/web_client

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Use polling to catch file changes from inside the container
# cf. https://stackoverflow.com/questions/44643045/running-development-server-with-create-react-app-inside-of-a-docker-container
ENV CHOKIDAR_USEPOLLING=true

# start the web client in development mode
CMD ["npm", "start"]

# "prod" is the production image
FROM dev as prod

# Copy sources within the image and build the web app
COPY . ./
RUN npm run build

# start image in production mode
CMD npx serve -s -l ${PORT} build
