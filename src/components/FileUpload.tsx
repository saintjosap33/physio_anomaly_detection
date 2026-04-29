import React, { useState } from "react";

type Props = {
  setResult: (data: any) => void;
};

export function FileUpload({ setResult }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    console.log("Clicked Analyze");

    // ❌ No file selected
    if (!file) {
      alert("Please select a CSV file first");
      return;
    }

    // ❌ Wrong file type
    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Only CSV files are allowed");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict-csv", {
        method: "POST",
        body: formData,
      });

      // ❌ Backend error handling
      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log("Response:", data);

      setResult(data);
    } catch (err: any) {
      console.error("Upload error:", err);

      setResult({
        status: "error",
        message: err.message || "Server not reachable",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel p-6 rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4">Upload CSV</h2>

      {/* File Input */}
      <input
        type="file"
        accept=".csv"
        className="block w-full text-sm text-gray-300 mb-3"
        onChange={(e) => {
          const selected = e.target.files?.[0] || null;
          console.log("Selected file:", selected);
          setFile(selected);
        }}
      />

      {/* Show selected file */}
      {file && (
        <p className="text-sm text-gray-400 mb-2">
          Selected: {file.name}
        </p>
      )}

      {/* Analyze Button */}
      <button
        onClick={upload}
        disabled={loading}
        className={`mt-2 px-4 py-2 rounded text-white transition ${
          loading
            ? "bg-gray-500 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 cursor-pointer"
        }`}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}