# syntax=docker/dockerfile:1.7
# ============================================
# Frontend Containerfile - Multi-stage Build
# Next.js standalone output → minimal runtime
# ============================================

# ============================================
# Stage 1: Base - Common dependencies
# ============================================
FROM docker.io/library/node:22-alpine3.21 AS base

RUN apk add --no-cache libc6-compat

WORKDIR /app

# ============================================
# Stage 2: Builder - Install deps & build
# ============================================
FROM base AS builder

COPY --from=docker.io/oven/bun:latest /usr/local/bin/bun /usr/local/bin/bun

COPY package.json bun.lock* ./

RUN --mount=type=cache,target=/root/.local/share/bun,sharing=locked \
    bun install --frozen-lockfile

COPY . .

ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-}

RUN bun run build

# ============================================
# Stage 3: Production (Alpine) - Secure runtime
# ============================================
FROM docker.io/library/node:22-alpine3.21 AS production

RUN apk add --no-cache libc6-compat

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

RUN addgroup --system --gid 1001 nextjs \
    && adduser --system --uid 1001 --ingroup nextjs --shell /usr/sbin/nologin nextjs

COPY --from=builder --chown=nextjs:nextjs /app/public ./public
COPY --from=builder --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nextjs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
