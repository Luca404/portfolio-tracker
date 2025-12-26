export default function AnalysisTabSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow p-6 animate-pulse">
      <div className="flex items-center gap-2 mb-6">
        <div className="w-6 h-6 bg-slate-200 rounded"></div>
        <div className="h-6 bg-slate-200 rounded w-64"></div>
      </div>
      <div className="h-4 bg-slate-200 rounded w-full mb-4"></div>
      <div className="h-4 bg-slate-200 rounded w-3/4 mb-8"></div>
      <div className="h-96 bg-slate-100 rounded"></div>
    </div>
  );
}
