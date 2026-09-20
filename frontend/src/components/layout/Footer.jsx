export default function Footer() {
  return (
    <footer className="border-t border-slate-800">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-6 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          ParkinsonAI — Explainable Multimodal AI Framework for Early Parkinson&apos;s
          Disease Risk Prediction
        </p>
        <p className="shrink-0">
          Research prototype · React + Vite + Express API
        </p>
      </div>
    </footer>
  );
}
