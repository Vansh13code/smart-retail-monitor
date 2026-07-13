import { createContext, useContext, useEffect, useState } from "react";

const UploadContext = createContext(null);
const LOCAL_STORAGE_KEY = "smart-retail-dashboard-upload";

const endpointLabels = {
  "/detect-shelf/": "Shelf Detection",
  "/detect-products/": "Product Detection",
  "/classify-products/": "Classification",
  "/inventory-analysis/": "Inventory Analysis",
  "/ocr/": "OCR",
  "/customer-analysis/": "Customer Analysis",
  "/complete-pipeline/": "Pipeline",
  "/generate-report/": "Reports",
  "/upload-image/": "Upload",
};

export function UploadProvider({ children }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [latestUpload, setLatestUpload] = useState(null);
  const [lastAnalysis, setLastAnalysis] = useState(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setLatestUpload(parsed.latestUpload || null);
        setLastAnalysis(parsed.lastAnalysis || null);
      }
    } catch (error) {
      console.warn("Unable to load upload state", error);
    }
  }, []);

  useEffect(() => {
    const payload = {
      latestUpload,
      lastAnalysis,
    };
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(payload));
  }, [latestUpload, lastAnalysis]);

  const selectFile = (file) => {
    const preview = file ? URL.createObjectURL(file) : null;
    setSelectedFile(file);
    setLatestUpload((current) => ({
      ...(current || {}),
      fileName: file?.name || current?.fileName,
      fileType: file?.type || current?.fileType,
      preview,
      selectedAt: new Date().toISOString(),
      fileSize: file?.size || current?.fileSize,
    }));
  };

  const registerUpload = (metadata, file) => {
    const preview = file ? URL.createObjectURL(file) : latestUpload?.preview || null;
    setSelectedFile(file || selectedFile);
    setLatestUpload({
      fileName: file?.name || metadata?.filename || latestUpload?.fileName,
      fileType: file?.type || metadata?.file_type || latestUpload?.fileType,
      path: metadata?.path || metadata?.filename || latestUpload?.path,
      metadata,
      preview,
      selectedAt: new Date().toISOString(),
    });
  };

  const registerAnalysis = (endpoint, result) => {
    setLastAnalysis({
      endpoint,
      title: endpointLabels[endpoint] || "Analysis",
      result,
      completedAt: new Date().toISOString(),
    });
  };

  const clearUpload = () => {
    setSelectedFile(null);
    setLatestUpload(null);
    setLastAnalysis(null);
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  };

  return (
    <UploadContext.Provider
      value={{
        selectedFile,
        latestUpload,
        lastAnalysis,
        selectFile,
        registerUpload,
        registerAnalysis,
        clearUpload,
      }}
    >
      {children}
    </UploadContext.Provider>
  );
}

export function useUploadContext() {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error("useUploadContext must be used within UploadProvider");
  }
  return context;
}
