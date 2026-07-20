import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG 知识库问答",
  description: "基于检索增强生成的智能问答系统",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
