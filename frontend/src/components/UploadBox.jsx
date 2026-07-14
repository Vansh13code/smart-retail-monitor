import { CloudUpload, FileText } from "lucide-react";

export default function UploadBox({ file, onFileChange, label }) {
  return (
    <label
      className="group block cursor-pointer rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center transition hover:border-blue-400 hover:bg-white"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        const nextFile = event.dataTransfer.files?.[0] || null;
        if (nextFile) onFileChange(nextFile);
      }}
    >
      <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-blue-100 text-blue-600 transition group-hover:bg-blue-200">
        <CloudUpload size={32} />
      </div>
      <p className="mt-4 text-lg font-semibold text-slate-900">{label || "Choose an image or video"}</p>
      <p className="mt-2 text-sm text-slate-500">Drag & drop or click to browse files</p>
      <input
        type="file"
        accept="image/*,video/*"
        className="sr-only"
        onChange={(event) => {
          const nextFile = event.target.files?.[0] || null;
          onFileChange(nextFile);
        }}
      />
      {file ? (
        <div className="mt-6 rounded-3xl bg-white p-4 text-left shadow-sm">
          <p className="font-semibold text-slate-900">Selected File</p>
          <p className="mt-1 text-sm text-slate-500">{file.name}</p>
          <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
      ) : (
        <div className="mt-6 inline-flex items-center gap-2 rounded-full bg-slate-100 px-4 py-2 text-sm text-slate-600">
          <FileText size={16} />
          Upload supported image/video files
        </div>
      )}
    </label>
  );
}
