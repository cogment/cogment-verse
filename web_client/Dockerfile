# "dev" is the developpment image
FROM node:14 as dev

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y bash git

WORKDIR /cogment-verse/web_client

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Code generation
COPY . ./
RUN npm run cogment_generate

# Use polling to catch file changes from inside the container
# cf. https://stackoverflow.com/questions/44643045/running-development-server-with-create-react-app-inside-of-a-docker-container
ENV CHOKIDAR_USEPOLLING=true

# start the web client in development mode
CMD ["npm", "start"]

# "prod" is the production image
FROM dev as prod

# Build the production version
RUN npm run build

# start image in production mode
CMD npx serve -s -l ${PORT} build