import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { UploadProvider } from "./contexts/UploadContext";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <UploadProvider>
      <App />
    </UploadProvider>
  </StrictMode>
);
