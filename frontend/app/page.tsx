import Link from "next/link";
import { BookOpen, MessageSquare } from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 mb-3">
            RAG 知识库问答
          </h1>
          <p className="text-slate-600">
            基于 FastAPI + Milvus + DeepSeek 的自研检索增强生成系统
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link
            href="/documents"
            className="group flex flex-col items-start p-8 bg-white rounded-xl border border-slate-200 hover:border-slate-900 hover:shadow-lg transition-all"
          >
            <BookOpen className="w-10 h-10 text-slate-900 mb-4 group-hover:scale-110 transition-transform" />
            <h2 className="text-xl font-semibold mb-2">文档管理</h2>
            <p className="text-sm text-slate-600">
              上传 PDF / Word / Markdown 文档,自动解析切分入库,管理已有文档
            </p>
          </Link>

          <Link
            href="/chat"
            className="group flex flex-col items-start p-8 bg-white rounded-xl border border-slate-200 hover:border-slate-900 hover:shadow-lg transition-all"
          >
            <MessageSquare className="w-10 h-10 text-slate-900 mb-4 group-hover:scale-110 transition-transform" />
            <h2 className="text-xl font-semibold mb-2">智能问答</h2>
            <p className="text-sm text-slate-600">
              基于知识库的对话问答,流式响应,引用溯源可点击查看原文
            </p>
          </Link>
        </div>

        <div className="mt-12 text-center text-xs text-slate-400">
          <p>M1 阶段:支持单文档上传 + 单轮问答</p>
        </div>
      </div>
    </main>
  );
}
