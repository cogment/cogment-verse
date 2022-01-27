# "dev" is the developpment image
#FROM node:14 as dev
FROM node:14-bullseye-slim as dev
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
