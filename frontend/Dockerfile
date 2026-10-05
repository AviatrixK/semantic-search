# Stage 1: build the static app with Node. Stage 2: serve it with Caddy (config is mounted by docker-compose.prod.yml).
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY . .
# Keep equal to the backend's MAX_UPLOAD_MB (client-side check only). No API URL: requests are relative (same origin).
ARG VITE_MAX_UPLOAD_MB=500
ENV VITE_MAX_UPLOAD_MB=$VITE_MAX_UPLOAD_MB
RUN npm run build

FROM caddy:2-alpine
COPY --from=build /app/dist /srv
