export default function ChartCard({ title, children }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
      </div>
      {children}
    </div>
  );
}
