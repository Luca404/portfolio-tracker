export default function PortfolioCardSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 relative animate-pulse">
      <div className="h-6 bg-slate-200 rounded w-3/4 mb-3"></div>
      <div className="h-4 bg-slate-200 rounded w-full mb-4"></div>
      <div className="flex items-center justify-between pt-4 border-t border-slate-200">
        <div className="h-4 bg-slate-200 rounded w-24"></div>
        <div className="h-6 bg-slate-200 rounded w-32"></div>
      </div>
    </div>
  );
}
