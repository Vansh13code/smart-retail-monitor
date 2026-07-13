import { useState } from "react";

export default function useApiRequest() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);

  const run = async (fn) => {
    setLoading(true);
    setError("");
    setProgress(0);

    try {
      const result = await fn((event) => {
        if (event.loaded && event.total) {
          setProgress(Math.round((event.loaded / event.total) * 100));
        }
      });
      setProgress(100);
      return result;
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    error,
    progress,
    run,
    setError,
  };
}
