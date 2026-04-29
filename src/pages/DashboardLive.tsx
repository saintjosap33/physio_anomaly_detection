import React, { useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { ResultPanel } from "@/components/ResultPanel";
import { Watch3D } from "@/components/Watch3D";

interface PredictionResult {
  status: string;
  anomaly?: boolean;
  score?: number;
  message?: string;
}

export function DashboardLive() {
  const [result, setResult] = useState<PredictionResult | null>(null);

  return (
    <div className="h-full w-full p-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">

        {/* LEFT: FILE UPLOAD */}
        <div className="lg:col-span-3 flex flex-col">
          <FileUpload setResult={setResult} />
        </div>

        {/* CENTER: WATCH */}
        <div className="lg:col-span-6 flex flex-col justify-center items-center">
          <div className="w-full h-full flex items-center justify-center">
            <Watch3D />
          </div>
        </div>

        {/* RIGHT: RESULT */}
        <div className="lg:col-span-3 flex flex-col">
          <ResultPanel result={result} />
        </div>

      </div>
    </div>
  );
}