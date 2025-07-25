FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 5173

# Run in development mode (for production, use a static server)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]