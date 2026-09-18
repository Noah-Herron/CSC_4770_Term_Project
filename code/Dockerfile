# Use Alpine Linux for the smallest possible footprint
FROM alpine:latest

# Install basic utilities (useful for debugging inside the node later)
RUN apk add --no-cache bash curl

# Set the working directory inside the container
WORKDIR /app

# (Optional) Copy your application code here later
# COPY . .

# Set a default environment variable for the node role
ENV NODE_ROLE=unassigned

# The command keeps the container alive and prints its role
CMD ["sh", "-c", "echo \"Starting $NODE_ROLE node...\" && tail -f /dev/null"]