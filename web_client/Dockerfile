
# pull official base image
FROM node:14 as dev

# set working directory
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

COPY . ./

# start app
CMD ["npm", "start"]
