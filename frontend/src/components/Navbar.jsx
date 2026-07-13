import { Bell, UserCircle } from "lucide-react";

export default function Navbar() {
  return (
    <header className="h-20 bg-white shadow flex items-center justify-between px-8">

      <div>
        <h1 className="text-3xl font-bold text-slate-800">
          Smart Retail Shelf Monitoring
        </h1>

        <p className="text-gray-500">
          AI Powered Retail Analytics Dashboard
        </p>
      </div>

      <div className="flex items-center gap-6">

        <button className="relative">

          <Bell size={24} className="text-slate-700"/>

          <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-red-500"></span>

        </button>

        <div className="flex items-center gap-3">

          <UserCircle size={40} className="text-blue-600"/>

          <div>

            <h2 className="font-semibold">
              Admin
            </h2>

            <p className="text-gray-500 text-sm">
              Smart Retail
            </p>

          </div>

        </div>

      </div>

    </header>
  );
}
