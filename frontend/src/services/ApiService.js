import axios from "axios";

const url = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: url,
  timeout: 300000,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === "ECONNABORTED") {
      error.message =
        "Analysis request timed out. Please try a smaller file or rerun after backend model warm-up.";
      return Promise.reject(error);
    }

    if (!error.response) {
      error.message =
        "Backend unavailable. Please verify that the Smart Retail API is running.";
    }
    return Promise.reject(error);
  }
);

function createFormData(file) {
  const form = new FormData();
  form.append("file", file);
  return form;
}

async function uploadFile(file, onUploadProgress) {
  const form = createFormData(file);
  const response = await api.post("/upload-image/", form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress,
  });
  return response.data;
}

async function postFile(endpoint, file, onUploadProgress) {
  const form = createFormData(file);
  const response = await api.post(endpoint, form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress,
  });
  return response.data;
}

async function postFilename(endpoint, filename) {
  const form = new FormData();
  form.append("filename", filename);
  const response = await api.post(endpoint, form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

async function pollVideoTask(taskId, onProgress, intervalMs = 2000) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const response = await api.get(`/video-analysis/status/${taskId}`);
        const task = response.data?.data || response.data;
        const progress = task.progress || 0;
        const status = task.status || "processing";

        if (onProgress) onProgress(progress, status);

        if (status === "completed") {
          clearInterval(timer);
          resolve(task.results || task);
        } else if (status === "failed" || status === "cancelled") {
          clearInterval(timer);
          reject(new Error(task.error || `Video processing ${status}.`));
        }
      } catch (err) {
        clearInterval(timer);
        reject(err);
      }
    }, intervalMs);
  });
}

async function healthCheck() {
  const response = await api.get("/health/");
  return response.data;
}

async function cancelVideoTask(taskId) {
  const response = await api.post(`/video-analysis/cancel/${taskId}`);
  return response.data;
}

export default {
  uploadFile,
  postFile,
  postFilename,
  pollVideoTask,
  cancelVideoTask,
  healthCheck,
};
