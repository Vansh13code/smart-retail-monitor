import { AlertTriangle } from "lucide-react";

export default function ErrorAlert({ message, onRetry }) {
  if (!message) return null;

  return (
    <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700 shadow-sm">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 text-red-600" />
        <div className="flex-1">
          <p className="font-semibold">Backend Offline</p>
          <p className="mt-1 whitespace-pre-wrap">{message}</p>
        </div>
      </div>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-full bg-red-600 px-4 py-2 text-white transition hover:bg-red-700"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
