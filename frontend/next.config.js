/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 反代后端 API,避免跨域
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
