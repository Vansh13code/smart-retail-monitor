export default function ResultCard({ title, data }) {
  if (!data) return null;

  return (
    <div className="rounded-[32px] border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <h2 className="text-2xl font-semibold text-slate-900">{title}</h2>
      </div>
      <div className="overflow-x-auto rounded-3xl bg-slate-950 p-5 text-left text-sm text-slate-100">
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  );
}
