import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import DashboardLayout from "../layouts/DashboardLayout";
import UploadBox from "../components/UploadBox";
import ImagePreview from "../components/ImagePreview";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";
import ResultCard from "../components/ResultCard";
import useUpload from "../hooks/useUpload";
import { useUploadContext } from "../contexts/UploadContext";

export default function Upload() {
  const { selectedFile, latestUpload, selectFile } = useUploadContext();
  const { loading, error, progress, upload } = useUpload();
  const [file, setFile] = useState(selectedFile);
  const [uploadResult, setUploadResult] = useState(null);

  useEffect(() => {
    if (!file && selectedFile) {
      setFile(selectedFile);
    }
  }, [selectedFile, file]);

  const previewUrl = useMemo(() => file?.preview || latestUpload?.preview || null, [file, latestUpload]);

  const handleFileChange = (nextFile) => {
    setUploadResult(null);
    setFile(nextFile);
    selectFile(nextFile);
  };

  const handleUpload = async () => {
    const nextFile = file || selectedFile;
    if (!nextFile) return;
    const result = await upload(nextFile);
    setUploadResult(result);
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div className="rounded-[32px] border border-slate-200 bg-white p-8 shadow-sm">
          <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Smart Retail Upload</p>
              <h1 className="mt-3 text-3xl font-semibold text-slate-900">Upload image or video for analysis</h1>
              <p className="mt-3 text-sm text-slate-500">Drag and drop a file or use the selector to upload and share it across all modules.</p>
            </div>
            <button
              type="button"
              onClick={handleUpload}
              disabled={!file || loading}
              className="inline-flex items-center gap-2 rounded-full bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? "Uploading..." : "Upload file"}
              <ArrowUpRight size={18} />
            </button>
          </div>

          <UploadBox file={file} onFileChange={handleFileChange} label="Upload an image or video" />

          {loading && (
            <div className="mt-8">
              <LoadingSpinner label={progress ? `Uploading ${progress}%` : "Uploading file"} />
            </div>
          )}

          {error && (
            <div className="mt-8">
              <ErrorAlert message={error} />
            </div>
          )}
        </div>

        <div className="grid gap-6 xl:grid-cols-[1fr_0.95fr]">
          <div className="rounded-[32px] border border-slate-200 bg-white p-8 shadow-sm">
            <h2 className="text-xl font-semibold text-slate-900">Upload summary</h2>
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <div className="rounded-3xl bg-slate-50 p-5">
                <p className="text-sm text-slate-500">Selected file</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{file?.name || latestUpload?.fileName || "No file selected"}</p>
              </div>
              <div className="rounded-3xl bg-slate-50 p-5">
                <p className="text-sm text-slate-500">File status</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{loading ? "Uploading" : uploadResult ? "Uploaded" : "Ready"}</p>
              </div>
              <div className="rounded-3xl bg-slate-50 p-5">
                <p className="text-sm text-slate-500">Latest upload</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{latestUpload?.fileName || "None"}</p>
              </div>
              <div className="rounded-3xl bg-slate-50 p-5">
                <p className="text-sm text-slate-500">Shared across modules</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{latestUpload ? "Yes" : "No"}</p>
              </div>
            </div>
          </div>

          <div className="rounded-[32px] border border-slate-200 bg-white p-8 shadow-sm">
            {previewUrl ? (
              <ImagePreview src={previewUrl} label="Selected file preview" />
            ) : (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-12 text-center text-slate-500">
                <p className="text-lg font-semibold">Preview unavailable</p>
                <p className="mt-2 text-sm">Choose a file to preview the selected image or video here.</p>
              </div>
            )}
          </div>
        </div>

        {uploadResult && <ResultCard title="Upload Result" data={uploadResult} />}
      </div>
    </DashboardLayout>
  );
}
