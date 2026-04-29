import React from "react";

type Props = {
  result: any;
};

export function ResultPanel({ result }: Props) {
  // 🟡 No result yet
  if (!result) {
    return (
      <div className="glass-panel p-6 rounded-xl mt-4 text-gray-400">
        Upload a CSV file and click Analyze to see results
      </div>
    );
  }

  // 🔴 Error state
  if (result.status === "error") {
    return (
      <div className="glass-panel p-6 rounded-xl mt-4 border border-red-500 text-red-400">
        <h2 className="text-lg font-bold mb-2">❌ Error</h2>
        <p>{result.message}</p>
      </div>
    );
  }

  const isAnomaly = result.anomaly === 1;

  return (
    <div className="glass-panel p-6 rounded-xl mt-4 shadow-lg">
      <h2 className="text-xl font-bold mb-4">ML Analysis Result</h2>

      {/* 🔥 Status */}
      <div className="mb-4">
        <p className="text-sm text-gray-400">Status</p>
        <p
          className={`text-lg font-semibold ${
            isAnomaly ? "text-red-500" : "text-green-500"
          }`}
        >
          {isAnomaly ? "🚨 Anomaly Detected" : "✅ Normal"}
        </p>
      </div>

      {/* 📊 Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-black/20 p-3 rounded-lg">
          <p className="text-sm text-gray-400">PCA Result</p>
          <p>{result.pca ?? "-"}</p>
        </div>

        <div className="bg-black/20 p-3 rounded-lg">
          <p className="text-sm text-gray-400">Isolation Forest</p>
          <p>{result.iso ?? "-"}</p>
        </div>

        <div className="bg-black/20 p-3 rounded-lg">
          <p className="text-sm text-gray-400">PCA Error</p>
          <p>
            {result.pca_error !== undefined
              ? result.pca_error.toFixed(6)
              : "-"}
          </p>
        </div>

        <div className="bg-black/20 p-3 rounded-lg">
          <p className="text-sm text-gray-400">Score</p>
          <p>
            {result.score !== undefined ? result.score.toFixed(4) : "-"}
          </p>
        </div>
      </div>
    </div>
  );
}