import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Upload,
  Boxes,
  Package,
  Tags,
  Warehouse,
  ScanLine,
  Users,
  Cpu,
  FileBarChart,
} from "lucide-react";

const menu = [
  { name: "Dashboard", icon: LayoutDashboard, path: "/" },
  { name: "Upload", icon: Upload, path: "/upload" },
  { name: "Shelf Detection", icon: Boxes, path: "/shelf" },
  { name: "Product Detection", icon: Package, path: "/product" },
  { name: "Classification", icon: Tags, path: "/classification" },
  { name: "Inventory", icon: Warehouse, path: "/inventory" },
  { name: "OCR", icon: ScanLine, path: "/ocr" },
  { name: "Customer", icon: Users, path: "/customer" },
  { name: "Pipeline", icon: Cpu, path: "/pipeline" },
  { name: "Reports", icon: FileBarChart, path: "/reports" },
];

export default function Sidebar() {
  return (
    <aside className="w-72 h-screen bg-slate-900 text-white fixed left-0 top-0">
      <div className="text-2xl font-bold p-6 border-b border-slate-700">
        Smart Retail
      </div>

      <nav className="mt-4 flex flex-col">
        {menu.map((item) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-6 py-4 hover:bg-slate-800 transition ${
                  isActive ? "bg-blue-600" : ""
                }`
              }
            >
              <Icon size={20} />
              {item.name}
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
