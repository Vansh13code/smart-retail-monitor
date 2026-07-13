export default function AnalyzeButton({ loading, onClick, title, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className="inline-flex items-center justify-center rounded-full bg-blue-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
    >
      {loading ? "Processing..." : title}
    </button>
  );
}
