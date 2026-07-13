export default function ImagePreview({ src, label, fileType }) {
  if (!src) return null;

  const isVideo = fileType?.startsWith("video/") || src?.endsWith(".mp4") || src?.endsWith(".webm");

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-6 py-4 bg-slate-50">
        <span className="text-sm font-semibold text-slate-700">{label}</span>
      </div>
      {isVideo ? (
        <video controls className="w-full bg-slate-950">
          <source src={src} />
        </video>
      ) : (
        <img className="w-full object-cover" src={src} alt={label} />
      )}
    </div>
  );
}
