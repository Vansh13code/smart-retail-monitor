import { useEffect, useMemo, useState, useCallback } from "react";
import { ArrowRight, RefreshCcw, FileText, CheckCircle, AlertTriangle, Image as ImageIcon, BarChart3, Clock } from "lucide-react";
import ApiService from "../services/ApiService";
import { useUploadContext } from "../contexts/UploadContext";
import useApiRequest from "../hooks/useApiRequest";
import ImagePreview from "./ImagePreview";
import LoadingSpinner from "./LoadingSpinner";
import ErrorAlert from "./ErrorAlert";
import ResultCard from "./ResultCard";
import UploadBox from "./UploadBox";

/* ─── helpers ─── */

function formatSummaryValue(value) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "string") return value.length > 60 ? `${value.slice(0, 60)}...` : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `Array(${value.length})`;
  if (typeof value === "object") return `Object(${Object.keys(value).length})`;
  return String(value);
}

function flatMetric(label, value, icon) {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-slate-50 px-4 py-3">
      <span className="text-blue-500">{icon}</span>
      <div>
        <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
        <p className="text-lg font-bold text-slate-900">{typeof value === "number" ? value : String(value ?? "N/A")}</p>
      </div>
    </div>
  );
}

/* ─── visual result renderers ─── */

function AnnotatedImageBlock({ src }) {
  if (!src) return null;
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="mb-3 text-sm uppercase tracking-[0.18em] text-slate-400">Annotated result</p>
      <img src={src} alt="Annotated" className="w-full rounded-2xl" />
    </div>
  );
}

function DetectionsTable({ detections }) {
  if (!detections?.length) return null;
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm overflow-x-auto">
      <p className="mb-4 text-sm uppercase tracking-[0.18em] text-slate-400">Detections</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-100 text-left text-slate-500">
            <th className="pb-2 pr-4">#</th>
            <th className="pb-2 pr-4">Label</th>
            <th className="pb-2 pr-4">Confidence</th>
            <th className="pb-2">Bounding Box</th>
          </tr>
        </thead>
        <tbody>
          {detections.slice(0, 50).map((d, i) => (
            <tr key={i} className="border-b border-slate-50 text-slate-700">
              <td className="py-2 pr-4 font-mono text-xs">{i + 1}</td>
              <td className="py-2 pr-4 font-semibold">{d.label || d.class || d.name || "object"}</td>
              <td className="py-2 pr-4">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-20 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full rounded-full bg-blue-500" style={{ width: `${((d.confidence || d.conf || 0) * 100).toFixed(0)}%` }} />
                  </div>
                  <span className="text-xs">{((d.confidence || d.conf || 0) * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="py-2 font-mono text-xs text-slate-500">
                {d.bbox ? JSON.stringify(d.bbox.map(v => Math.round(v))) : d.box ? JSON.stringify(d.box) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {detections.length > 50 && <p className="mt-2 text-xs text-slate-400">Showing first 50 of {detections.length}</p>}
    </div>
  );
}

function HeatmapBlock({ src }) {
  if (!src) return null;
  const imgSrc = src.startsWith("data:") ? src : `data:image/png;base64,${src}`;
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="mb-3 text-sm uppercase tracking-[0.18em] text-slate-400">Customer heatmap</p>
      <img src={imgSrc} alt="Heatmap" className="w-full rounded-2xl" />
    </div>
  );
}

function CategoryChart({ categories }) {
  if (!categories || !Object.keys(categories).length) return null;
  const entries = Object.entries(categories);
  const maxVal = Math.max(...entries.map(([, v]) => Number(v)), 1);
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <p className="mb-4 text-sm uppercase tracking-[0.18em] text-slate-400">Category breakdown</p>
      <div className="space-y-3">
        {entries.map(([name, value]) => (
          <div key={name} className="flex items-center gap-3">
            <span className="w-24 truncate text-sm font-medium text-slate-700">{name}</span>
            <div className="flex-1 h-5 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all" style={{ width: `${(Number(value) / maxVal) * 100}%` }} />
            </div>
            <span className="w-8 text-right text-sm font-bold text-slate-900">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VideoProgressBar({ progress, status, onCancel }) {
  return (
    <div className="rounded-3xl border border-blue-100 bg-blue-50 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-semibold text-blue-800">Video Processing</p>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-blue-600">{status} • {progress}%</span>
          {onCancel && status === "processing" && (
            <button
              type="button"
              onClick={onCancel}
              className="text-xs font-semibold text-red-600 hover:text-red-800 transition"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-blue-200">
        <div
          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

/* ─── main component ─── */

export default function AnalysisPage({ title, endpoint }) {
  const { selectedFile, latestUpload, selectFile, registerAnalysis, registerUpload } = useUploadContext();
  const [file, setFile] = useState(selectedFile);
  const [response, setResponse] = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);
  const { loading, error, progress, run, setError, setLoading } = useApiRequest();

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
  const hasRealFile = file instanceof File || selectedFile instanceof File;
  const hasFilename = !hasRealFile && !!(latestUpload?.fileName);
  const canAnalyze = hasRealFile || hasFilename;

  const handleFileChange = (nextFile) => {
    setResponse(null);
    setError("");
    setVideoProgress(null);
    selectFile(nextFile);
    setFile(nextFile);
  };

  const handleAnalyze = useCallback(async () => {
    setResponse(null);
    setVideoProgress(null);

    const fileToSend = file instanceof File ? file : (selectedFile instanceof File ? selectedFile : null);
    let uploadedFilename = latestUpload?.fileName;

    if (fileToSend) {
      const isAlreadyUploaded = latestUpload && 
        (latestUpload.fileName === fileToSend.name || latestUpload.metadata?.filename === fileToSend.name) &&
        (latestUpload.fileSize === fileToSend.size);

      if (!isAlreadyUploaded) {
        try {
          setLoading(true);
          setError("");
          const uploadResult = await run((event) => ApiService.uploadFile(fileToSend, event));
          if (uploadResult) {
            registerUpload(uploadResult, fileToSend);
            const data = uploadResult.data || uploadResult;
            uploadedFilename = data.filename;
          } else {
            throw new Error("File upload failed.");
          }
        } catch (err) {
          setError(err.message || "Failed to upload file.");
          setLoading(false);
          return;
        }
      }
    }

    if (uploadedFilename) {
      try {
        setLoading(true);
        setError("");
        const result = await ApiService.postFilename(endpoint, uploadedFilename);
        if (result) {
          const data = result.data || result;
          if (data?.task_id && data?.is_video) {
            setVideoProgress({ progress: 0, status: "processing", taskId: data.task_id });
            try {
              const videoResult = await ApiService.pollVideoTask(
                data.task_id,
                (prog, stat) => setVideoProgress({ progress: prog, status: stat, taskId: data.task_id })
              );
              setVideoProgress(null);
              setResponse(videoResult);
              registerAnalysis(endpoint, videoResult);
            } catch (pollErr) {
              setVideoProgress(null);
              setError(pollErr.message || "Video processing failed.");
            }
          } else {
            setResponse(result);
            registerAnalysis(endpoint, result);
          }
        }
      } catch (err) {
        setError(err.message || "Analysis failed.");
      } finally {
        setLoading(false);
      }
    } else {
      setError("Please select a file before running analysis.");
    }
  }, [file, selectedFile, latestUpload, endpoint, run, setError, setLoading, registerAnalysis, registerUpload]);

  /* ─── extract visual data from response ─── */
  const visualData = useMemo(() => {
    if (!response) return {};
    const root = response.data || response;
    const result = root.result || root;

    return {
      annotatedImage: root.annotated_image || result.annotated_image || null,
      detections: root.detections || result.detections || null,
      heatmap: root.heatmap || result.heatmap || null,
      totalDetections: root.total_detections ?? result.total_detections ?? null,
      totalCustomers: root.customer_count ?? root.total_customers ?? result.total_customers ?? result.customer_count ?? null,
      totalPriceTags: root.total_price_tags ?? result.total_price_tags ?? null,
      totalProducts: root.product_count ?? root.total_products ?? result.total_products ?? result.products?.total_products ?? null,
      categoryCount: result.category_count || result.classification?.category_count || result.class_count || null,
      occupancy: root.overall_occupancy ?? result.occupancy_percentage ?? null,
      emptyShelfPct: root.empty_shelf_percentage ?? null,
      lowStock: root.low_stock ?? null,
      misplacedProducts: root.misplaced_products ?? null,
      inventorySummary: root.summary ?? null,
      processingTime: response.processing_time ?? root.processing_time ?? null,
      warnings: root.warnings || result.warnings || [],
      productAnalysis: root.product_analysis || result.product_analysis || null,
      shelfInventory: result.shelf_inventory || result.products?.shelf_inventory || null,
      inventoryResult: Array.isArray(result) ? result : (Array.isArray(root.result) ? root.result : null),
      ocrText: root.ocr_text ?? null,
      priceList: Array.isArray(root.price) ? root.price : null,
      entryCount: root.entry_count ?? result.entry_count ?? null,
      exitCount: root.exit_count ?? result.exit_count ?? null,
      dwellTime: root.dwell_time ?? null,
      report: root.report || result.report || null,
      pdfUrl: root.pdf_url ?? null,
      annotatedVideoUrl: root.annotated_video_url ?? null,
      statistics: root.statistics ?? null,
      classes: root.classes ?? null,
      confidence: typeof root.confidence === "number" ? root.confidence : null,
    };
  }, [response]);

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
                disabled={loading || !canAnalyze}
                className="inline-flex items-center gap-2 rounded-full bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {loading ? "Analyzing..." : "Run Analysis"}
                <ArrowRight size={18} />
              </button>
              <button
                type="button"
                onClick={() => { setResponse(null); setVideoProgress(null); }}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300"
              >
                <RefreshCcw size={16} /> Reset
              </button>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.95fr]">
            <div className="space-y-6">
              <UploadBox file={file} onFileChange={handleFileChange} label="Select an image or video for analysis" />
              {videoProgress && (
                <VideoProgressBar
                  progress={videoProgress.progress}
                  status={videoProgress.status}
                  onCancel={async () => {
                    if (videoProgress.taskId) {
                      try {
                        await ApiService.cancelVideoTask(videoProgress.taskId);
                      } catch (cErr) {
                        console.warn("Cancellation request failed:", cErr);
                      }
                    }
                  }}
                />
              )}
              {loading && !videoProgress && <LoadingSpinner label={progress ? `Uploading ${progress}%` : "Running analysis"} />}
              {error && <ErrorAlert message={error} onRetry={handleAnalyze} />}
            </div>

            <div className="space-y-6">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Selected file</p>
                <p className="mt-3 text-lg font-semibold text-slate-900">{file?.name || latestUpload?.fileName || "No file selected"}</p>
                <p className="mt-2 text-sm text-slate-500">{file?.type || latestUpload?.fileType || "Upload an image or video to start."}</p>
                {hasFilename && !hasRealFile && (
                  <p className="mt-3 text-xs text-green-600 flex items-center gap-1"><CheckCircle size={12} /> Server file available — analysis enabled.</p>
                )}
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
                  {visualData.processingTime != null && (
                    <div className="flex items-center justify-between">
                      <span>Processing time</span>
                      <span className="font-semibold text-slate-900">{visualData.processingTime}s</span>
                    </div>
                  )}
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

      {/* ─── visual results ─── */}
      {response && (
        <div className="space-y-6">
          {/* warnings */}
          {visualData.warnings?.length > 0 && (
            <div className="rounded-3xl border border-amber-200 bg-amber-50 p-4 shadow-sm">
              <div className="flex items-center gap-2 text-amber-700">
                <AlertTriangle size={18} />
                <p className="text-sm font-semibold">Warnings</p>
              </div>
              <ul className="mt-2 space-y-1 text-sm text-amber-800">
                {visualData.warnings.map((w, i) => <li key={i}>• {w}</li>)}
              </ul>
            </div>
          )}

          {/* quick metric cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {visualData.totalDetections != null && flatMetric("Total Detections", visualData.totalDetections, <BarChart3 size={20} />)}
            {visualData.totalProducts != null && flatMetric("Products", visualData.totalProducts, <BarChart3 size={20} />)}
            {visualData.totalCustomers != null && flatMetric("Customers", visualData.totalCustomers, <BarChart3 size={20} />)}
            {visualData.totalPriceTags != null && flatMetric("Price Tags", visualData.totalPriceTags, <BarChart3 size={20} />)}
            {visualData.occupancy != null && flatMetric("Occupancy", `${visualData.occupancy}%`, <BarChart3 size={20} />)}
            {visualData.processingTime != null && flatMetric("Time", `${visualData.processingTime}s`, <Clock size={20} />)}
          </div>

          {/* annotated image */}
          <AnnotatedImageBlock src={visualData.annotatedImage} />

          {/* heatmap */}
          <HeatmapBlock src={visualData.heatmap} />

          {/* detections table */}
          <DetectionsTable detections={visualData.detections} />

          {/* category chart */}
          <CategoryChart categories={visualData.categoryCount} />

          {/* inventory results */}
          {visualData.inventoryResult && (
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm overflow-x-auto">
              <p className="mb-4 text-sm uppercase tracking-[0.18em] text-slate-400">Inventory report</p>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 text-left text-slate-500">
                    <th className="pb-2 pr-4">Shelf</th>
                    <th className="pb-2 pr-4">Products</th>
                    <th className="pb-2 pr-4">Occupancy</th>
                    <th className="pb-2 pr-4">Empty</th>
                    <th className="pb-2 pr-4">Low stock</th>
                    <th className="pb-2 pr-4">Out of stock</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {visualData.inventoryResult.map((shelf, i) => (
                    <tr key={i} className="border-b border-slate-50 text-slate-700">
                      <td className="py-2 pr-4 font-semibold">{shelf.shelf_id}</td>
                      <td className="py-2 pr-4">{shelf.product_count}</td>
                      <td className="py-2 pr-4">{shelf.occupancy_percentage}%</td>
                      <td className="py-2 pr-4">{shelf.empty_spaces}</td>
                      <td className="py-2 pr-4">{shelf.low_stock ? "⚠ Yes" : "No"}</td>
                      <td className="py-2 pr-4">{shelf.out_of_stock ? "❌ Yes" : "No"}</td>
                      <td className="py-2">
                        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${shelf.status === "Well-Stocked" ? "bg-green-100 text-green-700" : shelf.status === "Low Stock" ? "bg-amber-100 text-amber-700" : "bg-red-100 text-red-700"}`}>
                          {shelf.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── quick metric cards: extra fields ── */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {visualData.entryCount != null && flatMetric("Entries", visualData.entryCount, <BarChart3 size={20} />)}
            {visualData.exitCount != null && flatMetric("Exits", visualData.exitCount, <BarChart3 size={20} />)}
            {visualData.dwellTime != null && flatMetric("Avg Dwell", `${visualData.dwellTime}s`, <Clock size={20} />)}
            {visualData.occupancy != null && flatMetric("Occupancy", `${visualData.occupancy}%`, <BarChart3 size={20} />)}
            {visualData.emptyShelfPct != null && flatMetric("Empty Shelves", `${visualData.emptyShelfPct}%`, <BarChart3 size={20} />)}
            {visualData.confidence != null && flatMetric("Confidence", `${(visualData.confidence * 100).toFixed(1)}%`, <BarChart3 size={20} />)}
          </div>

          {/* ── inventory summary card ── */}
          {visualData.inventorySummary && (
            <div className="rounded-3xl border border-green-200 bg-green-50 p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-green-600 font-semibold mb-2">Inventory Summary</p>
              <p className="text-slate-700 text-sm leading-6">{visualData.inventorySummary}</p>
              {visualData.misplacedProducts?.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs font-semibold text-amber-700 mb-1">Misplaced Products:</p>
                  <div className="flex flex-wrap gap-2">
                    {visualData.misplacedProducts.map((p, i) => (
                      <span key={i} className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">{p}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── OCR text result ── */}
          {visualData.ocrText && (
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-slate-400 mb-4">OCR Result</p>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl bg-slate-50 p-4">
                  <p className="text-xs text-slate-500 mb-1">Detected Text</p>
                  <p className="text-lg font-bold text-slate-900 break-all">{visualData.ocrText}</p>
                </div>
                {visualData.priceList && visualData.priceList.length > 0 && (
                  <div className="rounded-2xl bg-blue-50 p-4">
                    <p className="text-xs text-blue-600 mb-1">Detected Price(s)</p>
                    {visualData.priceList.map((price, i) => (
                      <p key={i} className="text-2xl font-bold text-blue-700">₹{price}</p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── annotated video result ── */}
          {visualData.annotatedVideoUrl && (
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-slate-400 mb-4">Annotated Video</p>
              <video
                src={`http://localhost:8000${visualData.annotatedVideoUrl}`}
                controls
                className="w-full rounded-2xl"
                style={{ maxHeight: 400 }}
              />
              <a
                href={`http://localhost:8000${visualData.annotatedVideoUrl}`}
                download
                className="mt-4 inline-flex items-center gap-2 rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700"
              >
                <FileText size={16} /> Download Annotated Video
              </a>
            </div>
          )}

          {/* ── PDF download button for reports ── */}
          {visualData.pdfUrl && (
            <div className="rounded-3xl border border-purple-200 bg-purple-50 p-6 shadow-sm flex flex-col sm:flex-row sm:items-center gap-4">
              <div className="flex-1">
                <p className="text-sm uppercase tracking-[0.18em] text-purple-600 font-semibold">PDF Report Generated</p>
                <p className="mt-1 text-slate-600 text-sm">Your analysis report is ready to download as a PDF.</p>
              </div>
              <a
                href={`http://localhost:8000${visualData.pdfUrl}`}
                download
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full bg-purple-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-purple-700 whitespace-nowrap"
              >
                <FileText size={16} /> Download PDF
              </a>
            </div>
          )}

          {/* ── detected classes ── */}
          {visualData.classes && Object.keys(visualData.classes).length > 0 && (
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-slate-400 mb-4">Detected Classes</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(visualData.classes).map(([cls, count]) => (
                  <span key={cls} className="rounded-full bg-blue-100 px-3 py-1 text-sm font-semibold text-blue-800">
                    {cls}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Raw JSON is available for debugging but must never replace the visual result. */}
          <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
            <details className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <summary className="cursor-pointer text-sm font-semibold text-slate-700">Show raw API response (debug)</summary>
              <div className="mt-4 max-h-[520px] overflow-auto"><ResultCard title="Full API Response" data={response} /></div>
            </details>
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Result summary</h2>
              <div className="mt-4 grid gap-3">
                {Object.entries(response.data || response).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between gap-4 rounded-3xl bg-slate-50 px-4 py-3">
                    <span className="text-sm text-slate-500">{key.replace(/_/g, " ")}</span>
                    <span className="max-w-[55%] truncate text-right text-sm font-semibold text-slate-900">{formatSummaryValue(value)}</span>
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
        </div>
      )}
    </div>
  );
}
