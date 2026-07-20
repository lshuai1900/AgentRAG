"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Upload, Trash2, ArrowLeft, FileText, Loader2, CheckCircle, XCircle } from "lucide-react";

interface Doc {
  id: number;
  filename: string;
  file_type: string;
  file_size: number;
  chunk_count: number;
  status: string;
  error_msg: string | null;
  created_at: string;
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) throw new Error("获取文档列表失败");
      const data = await res.json();
      setDocs(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "上传失败");
      }
      await fetchDocs();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("确定删除该文档?关联的向量也会一起删除。")) return;
    try {
      const res = await fetch(`/api/documents/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("删除失败");
      await fetchDocs();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <main className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 className="text-xl font-semibold">文档管理</h1>
          </div>
          <Link
            href="/chat"
            className="text-sm text-slate-600 hover:text-slate-900"
          >
            去问答 →
          </Link>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* 上传区 */}
        <div className="mb-8">
          <label
            htmlFor="file-upload"
            className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
              uploading
                ? "border-slate-400 bg-slate-50"
                : "border-slate-300 hover:border-slate-900 hover:bg-slate-50"
            }`}
          >
            {uploading ? (
              <>
                <Loader2 className="w-8 h-8 text-slate-500 animate-spin mb-2" />
                <p className="text-sm text-slate-600">
                  正在上传与处理 (解析 → 切分 → embedding → 入库)...
                </p>
              </>
            ) : (
              <>
                <Upload className="w-8 h-8 text-slate-400 mb-2" />
                <p className="text-sm text-slate-600">
                  点击或拖拽文件上传
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  支持 PDF / Word (.docx) / Markdown
                </p>
              </>
            )}
            <input
              id="file-upload"
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.md,.markdown"
              className="hidden"
              onChange={handleUpload}
              disabled={uploading}
            />
          </label>
        </div>

        {error && (
          <div className="mb-6 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        {/* 文档列表 */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="text-sm font-semibold">
              文档列表 ({docs.length})
            </h2>
          </div>

          {loading ? (
            <div className="px-6 py-12 text-center text-sm text-slate-500">
              <Loader2 className="w-6 h-6 mx-auto animate-spin mb-2" />
              加载中...
            </div>
          ) : docs.length === 0 ? (
            <div className="px-6 py-12 text-center text-sm text-slate-500">
              暂无文档,请先上传
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {docs.map((doc) => (
                <div
                  key={doc.id}
                  className="px-6 py-4 flex items-center gap-4 hover:bg-slate-50"
                >
                  <div className="flex-shrink-0">
                    <FileText className="w-8 h-8 text-slate-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium truncate">
                        {doc.filename}
                      </span>
                      <StatusBadge status={doc.status} />
                    </div>
                    <div className="text-xs text-slate-500 flex items-center gap-3">
                      <span>{doc.file_type.toUpperCase()}</span>
                      <span>{formatSize(doc.file_size)}</span>
                      <span>{doc.chunk_count} chunks</span>
                      <span>{formatTime(doc.created_at)}</span>
                    </div>
                    {doc.error_msg && (
                      <div className="text-xs text-red-600 mt-1">
                        {doc.error_msg}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="删除"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { color: string; icon?: any }> = {
    ready: { color: "bg-green-100 text-green-700", icon: CheckCircle },
    pending: { color: "bg-slate-100 text-slate-600" },
    parsing: { color: "bg-blue-100 text-blue-700" },
    embedding: { color: "bg-blue-100 text-blue-700" },
    failed: { color: "bg-red-100 text-red-700", icon: XCircle },
  };
  const c = config[status] || config.pending;
  const Icon = c.icon;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs ${c.color}`}
    >
      {Icon && <Icon className="w-3 h-3" />}
      {status}
    </span>
  );
}
