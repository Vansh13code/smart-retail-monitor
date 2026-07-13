import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

export default function DashboardLayout({ children }) {

  return (

    <div className="flex bg-slate-100">

      <Sidebar/>

      <div className="ml-72 flex-1 min-h-screen">

        <Navbar/>

        <main className="p-8">

          {children}

        </main>

      </div>

    </div>

  );

}
