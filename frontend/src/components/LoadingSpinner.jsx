export default function LoadingSpinner({ size = 40, label = "Loading" }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <div
        className="h-10 w-10 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin"
        style={{ width: size, height: size }}
      ></div>
      <span className="text-sm text-slate-600">{label}</span>
    </div>
  );
}
