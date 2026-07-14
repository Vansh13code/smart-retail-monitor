import ApiService from "../services/ApiService";
import useApiRequest from "./useApiRequest";
import { useUploadContext } from "../contexts/UploadContext";

export default function useUpload() {
  const { selectFile, registerUpload } = useUploadContext();
  const { loading, error, progress, run, setError } = useApiRequest();

  const upload = async (file) => {
    if (!file) {
      throw new Error("Please choose a file before uploading.");
    }

    const response = await run((progressEvent) => ApiService.uploadFile(file, progressEvent));
    registerUpload(response, file);
    return response;
  };

  return {
    loading,
    error,
    progress,
    upload,
    selectFile,
    setError,
  };
}
