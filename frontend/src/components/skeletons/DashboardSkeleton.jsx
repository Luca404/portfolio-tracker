export default function DashboardSkeleton() {
  return (
    <div className="animate-pulse">
      {/* Header skeleton */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <div className="h-8 bg-slate-200 rounded w-64 mb-2"></div>
          <div className="h-4 bg-slate-200 rounded w-48"></div>
        </div>
        <div className="flex gap-3">
          <div className="h-10 bg-slate-200 rounded w-32"></div>
          <div className="h-10 bg-slate-200 rounded w-32"></div>
        </div>
      </div>

      {/* Metric cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 bg-slate-200 rounded-lg"></div>
              <div className="h-3 bg-slate-200 rounded w-24"></div>
            </div>
            <div className="h-8 bg-slate-200 rounded w-32"></div>
          </div>
        ))}
      </div>

      {/* Chart skeleton */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
        <div className="h-6 bg-slate-200 rounded w-48 mb-6"></div>
        <div className="h-64 bg-slate-100 rounded"></div>
      </div>
    </div>
  );
}
