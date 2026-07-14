export default function StatsCard({ title, value, delta, icon }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.24em] text-slate-500">{title}</p>
          <p className="mt-3 text-3xl font-semibold text-slate-900">{value}</p>
        </div>
        <div className="rounded-2xl bg-blue-50 p-3 text-blue-600">{icon}</div>
      </div>
      {delta ? (
        <p className="mt-3 text-sm text-slate-500">{delta}</p>
      ) : null}
    </div>
  );
}
