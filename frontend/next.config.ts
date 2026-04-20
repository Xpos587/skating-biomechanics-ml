import path from "node:path"
import type { NextConfig } from "next"
import createNextIntlPlugin from "next-intl/plugin"
import withBundleAnalyzer from "@next/bundle-analyzer"

const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: { root: path.resolve(__dirname) },
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://localhost:8000/api/:path*" }]
  },
}

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts")
export default withBundleAnalyzer({
  enabled: process.env.ANALYZE === "true",
})(withNextIntl(nextConfig))
