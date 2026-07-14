const LARGE_TEXT_LIMIT = 300;

function sanitizeForDisplay(value, key = "") {
  if (typeof value === "string") {
    const looksLikeImageData = value.startsWith("data:image/") || key.includes("image");
    if (looksLikeImageData) {
      return `[omitted image data: ${value.length} chars]`;
    }
    if (value.length > LARGE_TEXT_LIMIT) {
      return `${value.slice(0, LARGE_TEXT_LIMIT)}... [truncated ${value.length - LARGE_TEXT_LIMIT} chars]`;
    }
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((item) => sanitizeForDisplay(item));
  }

  if (value && typeof value === "object") {
    const output = {};
    for (const [childKey, childValue] of Object.entries(value)) {
      output[childKey] = sanitizeForDisplay(childValue, childKey.toLowerCase());
    }
    return output;
  }

  return value;
}

export default function ResultCard({ title, data }) {
  if (!data) return null;

  const displayData = sanitizeForDisplay(data);

  return (
    <div className="rounded-[32px] border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <h2 className="text-2xl font-semibold text-slate-900">{title}</h2>
      </div>
      <div className="overflow-x-auto rounded-3xl bg-slate-950 p-5 text-left text-sm text-slate-100">
        <pre>{JSON.stringify(displayData, null, 2)}</pre>
      </div>
    </div>
  );
}
