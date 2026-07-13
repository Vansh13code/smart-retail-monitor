import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Shelf from "./pages/Shelf";
import Product from "./pages/Product";
import Classification from "./pages/Classification";
import Inventory from "./pages/Inventory";
import OCR from "./pages/OCR";
import Customer from "./pages/Customer";
import Pipeline from "./pages/Pipeline";
import Reports from "./pages/Reports";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/shelf" element={<Shelf />} />
        <Route path="/product" element={<Product />} />
        <Route path="/classification" element={<Classification />} />
        <Route path="/inventory" element={<Inventory />} />
        <Route path="/ocr" element={<OCR />} />
        <Route path="/customer" element={<Customer />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/reports" element={<Reports />} />
      </Routes>
    </BrowserRouter>
  );
}
