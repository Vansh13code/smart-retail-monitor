import { useEffect, useMemo, useState } from "react";
import { ArrowRight, RefreshCcw, FileText } from "lucide-react";
import ApiService from "../services/ApiService";
import { useUploadContext } from "../contexts/UploadContext";
import useApiRequest from "../hooks/useApiRequest";
import ImagePreview from "./ImagePreview";
import LoadingSpinner from "./LoadingSpinner";
import ErrorAlert from "./ErrorAlert";
import ResultCard from "./ResultCard";
import UploadBox from "./UploadBox";

export default function AnalysisPage({ title, endpoint }) {
  const { selectedFile, latestUpload, selectFile, registerAnalysis } = useUploadContext();
  const [file, setFile] = useState(selectedFile);
  const [response, setResponse] = useState(null);
  const { loading, error, progress, run } = useApiRequest();

  useEffect(() => {
    if (!file && selectedFile) {
      setFile(selectedFile);
    }
  }, [selectedFile, file]);

  useEffect(() => {
    if (!file && latestUpload?.preview) {
      setFile({ name: latestUpload.fileName, preview: latestUpload.preview, type: latestUpload.fileType });
    }
  }, [latestUpload, file]);

  const previewUrl = useMemo(() => file?.preview || latestUpload?.preview || null, [file, latestUpload]);

  const handleFileChange = (nextFile) => {
    setResponse(null);
    selectFile(nextFile);
    setFile(nextFile);
  };

  const handleAnalyze = async () => {
    const fileToSend = file instanceof File ? file : selectedFile;

    if (!fileToSend) {
      return;
    }

    const result = await run((event) => ApiService.postFile(endpoint, fileToSend, event));
    if (result) {
      setResponse(result);
      registerAnalysis(endpoint, result);
    }
  };

  return (
    <div className="space-y-8">
      <div className="grid gap-6 xl:grid-cols-[1.65fr_1fr]">
        <div className="rounded-[32px] border border-slate-200 bg-white p-8 shadow-sm">
          <div className="mb-6 flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">{title}</p>
              <h1 className="mt-3 text-3xl font-semibold text-slate-900">Advanced {title}</h1>
              <p className="mt-3 text-sm text-slate-500">Run the backend analysis on your selected file and review rich results instantly.</p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="inline-flex items-center gap-2 rounded-full bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {loading ? "Analyzing..." : "Run Analysis"}
                <ArrowRight size={18} />
              </button>
              <button
                type="button"
                onClick={() => setResponse(null)}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300"
              >
                <RefreshCcw size={16} /> Reset
              </button>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.95fr]">
            <div className="space-y-6">
              <UploadBox file={file} onFileChange={handleFileChange} label="Select an image or video for analysis" />
              {loading && <LoadingSpinner label={progress ? `Uploading ${progress}%` : "Running analysis"} />}
              {error && <ErrorAlert message={error} onRetry={handleAnalyze} />}
            </div>

            <div className="space-y-6">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Selected file</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{file?.name || latestUpload?.fileName || "No file selected"}</p>
                <p className="mt-2 text-sm text-slate-500">{file?.type || latestUpload?.fileType || "Upload an image or video to start."}</p>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Analysis info</p>
                <div className="mt-4 space-y-3 text-sm text-slate-600">
                  <div className="flex items-center justify-between">
                    <span>Endpoint</span>
                    <span className="font-semibold text-slate-900">{endpoint}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Last result</span>
                    <span className="font-semibold text-slate-900">{response ? "Completed" : "Pending"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Upload shared</span>
                    <span className="font-semibold text-slate-900">{latestUpload ? "Yes" : "No"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {previewUrl ? (
            <ImagePreview src={previewUrl} label="File preview" fileType={file?.type || latestUpload?.fileType} />
          ) : (
            <div className="rounded-3xl border border-slate-200 bg-white p-12 text-center text-slate-500 shadow-sm">
              <p className="text-lg font-semibold">No preview available</p>
              <p className="mt-2 text-sm">Upload a file to see the selected image or video preview here.</p>
            </div>
          )}
        </div>
      </div>

      {response && (
        <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
          <div className="space-y-6">
            <ResultCard title="Full API Response" data={response} />
          </div>
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-slate-900">Result summary</h2>
            <div className="mt-4 grid gap-3">
              {Object.entries(response).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between gap-4 rounded-3xl bg-slate-50 px-4 py-3">
                  <span className="text-sm text-slate-500">{key.replace(/_/g, " ")}</span>
                  <span className="max-w-[55%] truncate text-right text-sm font-semibold text-slate-900">{typeof value === "object" ? JSON.stringify(value).slice(0, 40) : String(value)}</span>
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                const blob = new Blob([JSON.stringify(response, null, 2)], { type: "application/json" });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                link.download = `${title.replace(/\s+/g, "_").toLowerCase()}_response.json`;
                link.click();
              }}
              className="mt-6 inline-flex items-center gap-2 rounded-full bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700"
            >
              <FileText size={16} /> Download JSON
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
