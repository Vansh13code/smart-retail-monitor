import { useEffect, useMemo, useState } from "react";
import { AreaChart, Area, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Database, Layers, Tags, Users, Cpu, Box } from "lucide-react";
import DashboardLayout from "../layouts/DashboardLayout";
import StatsCard from "../components/StatsCard";
import ChartCard from "../components/ChartCard";
import ApiService from "../services/ApiService";
import { useUploadContext } from "../contexts/UploadContext";

export default function Dashboard() {
  const { latestUpload, lastAnalysis } = useUploadContext();
  const [health, setHealth] = useState({ status: "Loading", message: "Checking backend status..." });
  const [online, setOnline] = useState(false);

  useEffect(() => {
    ApiService.healthCheck()
      .then((data) => {
        setHealth(data);
        setOnline(true);
      })
      .catch((error) => {
        setHealth({ status: "offline", message: error.message });
        setOnline(false);
      });
  }, []);

  const summary = useMemo(() => {
    const data = lastAnalysis?.result || {};
    return {
      products: data.products?.total_products ?? 0,
      shelves: data.products?.shelf_inventory?.length ?? data.detections?.length ?? 0,
      ocrTags: data.ocr?.total_price_tags ?? data.total_price_tags ?? 0,
      customers: data.customers?.total_customers ?? 0,
      inventory: data.inventory?.length ?? 0,
      lastEndpoint: lastAnalysis?.title ?? "No run yet",
    };
  }, [lastAnalysis]);

  const chartData = useMemo(() => {
    const categories = lastAnalysis?.result?.classification?.category_count ?? {};
    const entries = Object.entries(categories);
    if (!entries.length) return [{ name: "Awaiting", value: 0 }];
    return entries.map(([name, value]) => ({ name, value: Number(value) }));
  }, [lastAnalysis]);

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div className="rounded-[32px] border border-slate-200 bg-white p-8 shadow-sm">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Overview</p>
              <h1 className="mt-3 text-3xl font-semibold text-slate-900">Smart Retail Shelf Monitoring</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">Monitor system health, recent analysis results, and module performance from a single dashboard.</p>
            </div>
            <div className="rounded-3xl bg-slate-50 px-6 py-4 text-sm text-slate-700 shadow-sm">
              <p className="font-semibold">Backend status</p>
              <p className={`mt-2 ${online ? "text-green-600" : "text-red-600"}`}>{online ? "Online" : "Offline"}</p>
              <p className="mt-1 text-xs text-slate-500">{health.message}</p>
            </div>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-3">
          <StatsCard title="Products" value={summary.products} icon={<Box size={24} />} delta={lastAnalysis ? `Last run: ${lastAnalysis.title}` : "No analysis yet"} />
          <StatsCard title="Shelves" value={summary.shelves} icon={<Layers size={24} />} delta={lastAnalysis ? `Endpoint: ${lastAnalysis.endpoint}` : "Awaiting first run"} />
          <StatsCard title="OCR Tags" value={summary.ocrTags} icon={<Tags size={24} />} delta={lastAnalysis ? `${summary.ocrTags} tags detected` : "Awaiting data"} />
          <StatsCard title="Customers" value={summary.customers} icon={<Users size={24} />} delta={lastAnalysis ? `Last run: ${lastAnalysis.title}` : "No customer data"} />
          <StatsCard title="Inventory" value={summary.inventory} icon={<Database size={24} />} delta={lastAnalysis ? `Shelves: ${summary.shelves}` : "No inventory result"} />
          <StatsCard title="Pipeline" value={lastAnalysis ? "Active" : "Idle"} icon={<Cpu size={24} />} delta={lastAnalysis ? `Last: ${summary.lastEndpoint}` : "Awaiting pipeline"} />
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.3fr_0.9fr]">
          <ChartCard title="Category distribution">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="categoryGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fill: "#475569", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#475569", fontSize: 12 }} />
                  <Tooltip />
                  <Area type="monotone" dataKey="value" stroke="#0ea5e9" fill="url(#categoryGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
          <div className="space-y-6">
            <div className="rounded-[32px] border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Latest upload</p>
              <p className="mt-4 text-xl font-semibold text-slate-900">{latestUpload?.fileName || "No upload available"}</p>
              <p className="mt-3 text-sm text-slate-500">{latestUpload ? `Uploaded on ${new Date(latestUpload.selectedAt).toLocaleString()}` : "Upload a file to begin analysis."}</p>
            </div>
            <div className="rounded-[32px] border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Recent activity</p>
              <div className="mt-5 grid gap-3">
                <div className="rounded-3xl bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-500">Last module</span>
                    <span className="font-semibold text-slate-900">{lastAnalysis?.title || "None"}</span>
                  </div>
                </div>
                <div className="rounded-3xl bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-500">Backend status</span>
                    <span className="font-semibold text-slate-900">{health.status}</span>
                  </div>
                </div>
                <div className="rounded-3xl bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-500">Upload shared</span>
                    <span className="font-semibold text-slate-900">{latestUpload ? "Yes" : "No"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
